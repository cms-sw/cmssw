#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <utility>

#include "TPRegexp.h"
#include "TEveManager.h"
#include "TEveScene.h"
#include "TEveElement.h"
#include "TEveGeoNode.h"
#include "TGeoNode.h"
#include "TCollection.h"
#include "TObjString.h"

#include "split.h"
#include "eve_filter.h"
#include "eve_macros.h"

// global variables... FIXME all this stuff should be incapsulated ina a class !
TPRegexp parents;
std::vector<TPRegexp> filters;
std::vector<Color_t>  colors;
unsigned int matching_nodes;

void split_path(const std::string & path, std::string & name, std::vector<std::string> & parents) {
  split( path, parents, '/', false );
  if (parents.empty()) {
    name = "";
  } else {
    name = parents.back();
    parents.pop_back();
  }
}

static TPRegexp ns_name_index( "([[:alnum:]]+:[[:alnum:]-\\[\\]]+)(_[0-9]+)+" );

// generalizes a node name of the form namespace:Name_NUM into a (regexp) filter that allows any number
TPRegexp make_filter(const std::string & token) {
  std::string filter;
  filter += "(";
  if (ns_name_index.MatchB( token ))
    filter += ((TObjString*)(ns_name_index.MatchS( token )->At(1)))->GetString();
  else
    filter += token;
  filter += ")_[0-9]+";
  return TPRegexp( filter.c_str() );
}

// generalizes a list of node names of the form namespace:Name_NUM into a (regexp) filter that allows any of the "namespace:Name" followd by any number
TPRegexp make_filter(const std::vector<std::string> & tokens) {
  if (tokens.empty())
    return TPRegexp();

  std::string filter;
  filter += "(";
  if (ns_name_index.MatchB( tokens[0] ))
    filter += ((TObjString*)(ns_name_index.MatchS( tokens[0] )->At(1)))->GetString();
  else
    filter += tokens[0];
  for (unsigned int i = 1; i < tokens.size(); ++i) {
    filter += "|";
    if (ns_name_index.MatchB( tokens[i] ))
      filter += ((TObjString*)(ns_name_index.MatchS( tokens[i] )->At(1)))->GetString();
    else
      filter += tokens[i];
  }
  filter += ")_[0-9]+";
  return TPRegexp( filter.c_str() );
}

// apply the filters to a node, specifying if the removed elements should be hidden (default), deleted (does not work) or ignored
void node_filter(TEveElement * node, int simplify /* = do_hide */, bool verbose /* = false */) {
  static int indent = 0;
  ++indent;

  expand_node(node);

  for (TEveElement::List_i i = node->BeginChildren(); i != node->EndChildren(); ++i)
  {
    bool found = false;
    TEveGeoNode * child = (TEveGeoNode*)(*i);
    for (unsigned int det = 0; det < filters.size(); ++det) {
      if (filters[det].MatchB( child->GetName() )) {
        // found a selcted leaf
        if (verbose) {
          for (int _t = 0; _t < indent; ++_t) std::cout << "  ";
          std::cout << child->GetName() << " (found)" << std::endl;
        }
        child->SetRnrSelf( true );
        child->SetMainColor( colors[det] );
        // simplify - hide or delete its children
        switch (simplify) {
          case do_hide:
            child->SetRnrSelf( true );
            child->SetRnrChildren( false );
            break;
          case do_remove:
            child->Destroy();
            break;
          case do_nothing:
            break;
        }

        found = true;
        ++matching_nodes;
        break;  // breaks out of the loop on "det"
      }
    }
    if (found)
      continue;
    else if (parents.MatchB( child->GetName() )) {
      // found a possible parent
      if (verbose) {
        for (int _t = 0; _t < indent; ++_t) std::cout << "  ";
        std::cout << child->GetName() << " (look inside...)" << std::endl;
      }
      child->SetRnrSelf( false );
      child->SetRnrChildren( true );
      node_filter( child );
    } else {
      // enything else
      if (verbose) {
        for (int _t = 0; _t < indent; ++_t) std::cout << "  ";
        std::cout << child->GetName() << " (unused)" << std::endl;
      }
      // simplify - hide or delete this
      switch (simplify) {
        case do_hide:
          child->SetRnrSelf( false );
          child->SetRnrChildren( false );
          break;
        case do_remove:
          child->Destroy();
          break;
        case do_nothing:
          break;
      }
    }
  }

  --indent;
}

// initializes the filter static variables from a list of elements to display and colors to use
void init_filter(const std::vector< std::pair< std::string, Color_t> > & elements) {
  std::vector< std::string > all_parents;
  std::vector< std::string > all_names;

  for (unsigned int i = 0; i < elements.size(); ++i) {
    std::vector< std::string > s_parents;
    std::string s_name;
    split_path( elements[i].first, s_name, s_parents );
    all_names.push_back( s_name );
    all_parents.insert( all_parents.end(), s_parents.begin(), s_parents.end() );

    colors.push_back( elements[i].second );
  }

  parents = make_filter( all_parents );
  for (unsigned int i = 0; i < all_names.size(); ++i)
    filters.push_back( make_filter( all_names[i] ) );

  matching_nodes = 0;
}

// dump the filters
void dump(void) {
  std::cout << parents.fPattern << std::endl;
  for (unsigned int i = 0; i < filters.size(); ++i)
    std::cout << '\t' << std::setw(32) << std::left << filters[i].fPattern << '\t' << colors[i] << std::endl;
}

// apply the filters to a node, then notify it to update itself and its children for redrawing
void apply_filter(TEveElement * node, int simplify /* = do_hide */, bool verbose /* = false */) {
  if (node == 0)
    return;

  if (verbose)
    std::cout << get_name(node) << " (look inside...)" << std::endl;

  node_filter( node, simplify );
  node->ElementChanged( true, true );

  std::cout << "found " << matching_nodes << " matching nodes" << std::endl;
}

// find a TEve root object (could be a TEveGeoRootNode or a TEveGeoShape) by its name
TEveElement * get_root_object(const char* name)
{
  for (TEveElement::List_i i = gEve->GetScenes()->BeginChildren(); i != gEve->GetScenes()->EndChildren(); ++i) {
    TEveScene * scene = dynamic_cast<TEveScene *> (*i);
    if (not scene)
      continue;
    for (TEveElement::List_i n = scene->BeginChildren(); n != scene->EndChildren(); ++n) {
      const char* obj_name = get_name( *n );
      if (obj_name == 0 or strcmp(name, obj_name))
        continue;

      return *n;
    }
  }

  return 0;
}
