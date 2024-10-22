#include <vector>
#include <list>
#include <string>
#include <cmath>

#include <TEveElement.h>
#include <TEveGeoNode.h>
#include <TGeoNode.h>

#include "eve_macros.h"

// get the name from an object derived from both TEveElement and TNamed
const char* get_name( const TEveElement * element ) {
  // try as a TEveGeoNode or TEveGeoShape
  if (const TEveGeoNode * node = dynamic_cast<const TEveGeoNode *>( element ))
    return node->GetName();
  if (const TEveGeoShape * shape = dynamic_cast<const TEveGeoShape *>( element ))
    return shape->GetName();

  // try to access the element as a generic named object
  if (const TNamed * named = dynamic_cast<const TNamed *>( element ))
    return named->GetName();

  return 0;
}

// get the title from an object derived from both TEveElement and TNamed
const char* get_title( const TEveElement * element ) {
  // try as a TEveGeoNode or TEveGeoShape
  if (const TEveGeoNode * node = dynamic_cast<const TEveGeoNode *>( element ))
    return node->GetTitle();
  if (const TEveGeoShape * shape = dynamic_cast<const TEveGeoShape *>( element ))
    return shape->GetTitle();

  // try to access the element as a generic named object
  if (const TNamed * named = dynamic_cast<const TNamed *>( element ))
    return named->GetTitle();

  return 0;
}

// force a node to expand its internal reprsentation, so all children are actually present
void expand_node( TEveElement * element )
{
  // force a TEveGeoNode to load all its children
  if (TEveGeoNode * node = dynamic_cast<TEveGeoNode *>( element )) {
    if (node->GetNChildren() == 0 && node->GetNode()->GetVolume()->GetNdaughters() > 0) {
      TIter next(node->GetNode()->GetVolume()->GetNodes());
      TGeoNode* dnode;
      while ((dnode = (TGeoNode*) next()) != 0) {
        TEveGeoNode* node_re = new TEveGeoNode(dnode);
        node->AddElement(node_re);
      }
    }
    return;
  }
  // a TEveGeoShape is always exanded
  //if (TEveGeoShape * shape __attribute__ ((unused)) = dynamic_cast<TEveGeoShape *>( element )) {
  //  return;
  //}
  // a generic TEveElement has no knwledge on children expansion
  return;
}

// retrieves a TShape from a TEveElement
const TGeoShape * get_shape( const TEveElement * element ) {
  // a TEveGeoNode, can look into its TGeoNode and retrieve the shape
  if (const TEveGeoNode * node = dynamic_cast<const TEveGeoNode *>( element )) {
    return node->GetNode()->GetVolume()->GetShape();
  }
  // a TEveGeoShape owns its shape
  if (const TEveGeoShape * shape = dynamic_cast<const TEveGeoShape *>( element )) {
    TEveGeoShape * nc_shape = const_cast<TEveGeoShape *>( shape );
    return const_cast<const TGeoShape *>( nc_shape->GetShape() );
  }
  // a TEveElement is too generic, no way to get a shape
  return 0;
}

// overloaded non-const TShape retrieval, allowed from a TGeoShape only
TGeoShape * get_shape( TEveElement * element ) {
  // a TEveGeoNode cannot modify its shape
  //if (const TEveGeoNode * node __attribute__ ((unused)) = dynamic_cast<const TEveGeoNode *>( element )) {
  //  return 0;
  //}
  // a TEveGeoShape owns its shape, and can modifiy it
  if (TEveGeoShape * shape = dynamic_cast<TEveGeoShape *>( element )) {
    return shape->GetShape();
  }
  // a TEveElement is too generic, no way to get a shape
  return 0;
}

// return a copy of the local-to-global transformation applied to a TEveElement
TGeoMatrix * get_transform( const TEveElement * element ) {
  if (const TEveGeoNode * node = dynamic_cast<const TEveGeoNode *>( element )) {
    // a TEveGeoNode is a proxy to a TGeoNode, which knows its relative transformation wrt. its parent
    // so we follow the TEveGeoNode hierarchy up to a TEveGeoTopNode, then jump to its TGeoManager, and go back down the branches to the TEveGeoNode's TGeoNode
    std::vector< const TEveGeoNode * > nodes;
    const TEveGeoTopNode * top = 0;

    while ((top = dynamic_cast<const TEveGeoTopNode *>( node )) == 0) {
      // save the current node
      nodes.push_back(node);

      // check that the node actually has any parents
      TEveGeoNode * nc_node = const_cast<TEveGeoNode *>( node );
      if (nc_node->BeginParents() == nc_node->EndParents())
        return 0;

      // assume the firt parent is the good one, and check that the parent type is correct
      node = dynamic_cast<const TEveGeoNode *>( * nc_node->BeginParents() );
      if (node == 0)
        return 0;
    }
    // reached the top level node, start from its (optional) global transormation
    TGeoHMatrix * matrix = new TGeoHMatrix();
    (const_cast<TEveGeoTopNode *>(top))->RefGlobalTrans().SetGeoHMatrix( *matrix );
    for (unsigned int i = 0; i < nodes.size(); ++i)
      *matrix *= *(nodes[i]->GetNode()->GetMatrix());
    return matrix;
  }

  if (const TEveGeoShape * shape = dynamic_cast<const TEveGeoShape *>( element )) {
    // a TEveGeoShape knows the absolute transformation of its shape
    TGeoHMatrix * matrix = new TGeoHMatrix();
    (const_cast<TEveGeoShape *>(shape))->RefHMTrans().SetGeoHMatrix( *matrix );
    return matrix;
  }

  return 0;
}

// clone a TEveGeoShape or TEveGeoNode into a new TEveGeoShape, and add it as a child to a parent if one is given
TEveGeoShape * clone( const TEveElement * element, TEveElement * parent /* = 0 */)
{
  TEveGeoShape* shape = new TEveGeoShape( get_name(element), get_title(element) );

  std::unique_ptr<TGeoMatrix> matrix( get_transform(element) );
  shape->SetTransMatrix( matrix.get() );
  delete matrix;

  TEveGeoShapeExtract extract;                  // FIXME put name and title here...
  extract.SetShape( (TGeoShape *) get_shape(element)->Clone() );
  extract.SetTrans( trans.Array() );
  extract.SetRnrSelf( true );
  extract.SetRnrElements( true );

  TEveGeoShape * clone = TEveGeoShape::ImportShapeExtract( &extract, parent );
  return clone;
}

// set an element's color and alpha, and possibly its children's up to levels levels deep
void set_color( TEveElement * element, Color_t color, float alpha /* = 1.0 */, unsigned int levels /* = 0 */)
{
  if (not element)
    return;

  // set this node's color
  element->SetMainColor( color );
  if (alpha > 1.) alpha = 1.;
  if (alpha < 0.) alpha = 0.;
  unsigned char transparency = (unsigned char) roundf(100. - (alpha * 100.));
  element->SetMainTransparency( transparency );

  if (levels > 0) {
    // set the node's children's color
    expand_node( element );
    for (std::list<TEveElement*>::iterator i = element->BeginChildren(); i != element->EndChildren(); ++i)
      set_color( *i, color, alpha, levels - 1);
  }
  // notify the element that it has changed
  element->ElementChanged(true, true);
}

// check if a node has any children or if it's a leaf node
bool is_leaf_node( const TEveElement * element )
{
  // a TEveGeoNode can have unaccounted-for children
  if (const TEveGeoNode * node = dynamic_cast<const TEveGeoNode *>( element )) {
    return ((node->GetNChildren() == 0) and (node->GetNode()->GetVolume()->GetNdaughters() == 0));
  }
  // a TEveGeoShape always knows its children
  if (const TEveGeoShape * shape = dynamic_cast<const TEveGeoShape *>( element )) {
    return (shape->GetNChildren() == 0);
  }
  // default implementation
  return (element->GetNChildren() == 0);
}

// toggle an elements's children visibility, based on their name
// names are checked only up to their length, so for example tec:TEC will match both tec:TEC_1 and tec:TEC_2
void set_children_visibility( TEveElement * element, const std::string & node_name, const std::vector<std::string> & children_name, bool visibility )
{
  // try to access the element as a named thingy
  const char * name = get_name( element );
  if (not name or strncmp(name, node_name.c_str(), node_name.size()))
    // unnamed node, or wrong node
    return;

  for (std::list<TEveElement *>::iterator j = element->BeginChildren(); j != element->EndChildren(); ++j) {
    TEveElement * child = *j;
    name = get_name( child );
    if (not name)
      // unnamed node, ignore it
      continue;

    for (unsigned int i = 0; i < children_name.size(); ++i)
      if (not strncmp(name, children_name[i].c_str(), children_name[i].size())) {
        // change this child visibility
        if (is_leaf_node( child )) {
          child->SetRnrSelf( visibility );
          child->SetRnrChildren( false );
        } else {
          child->SetRnrSelf( false );
          child->SetRnrChildren( visibility );
        }
        break;
      }
  }
  // notify the element that is had changed
  element->ElementChanged(true, true);
}

// set Tracker's Endcaps visibility
void set_tracker_endcap_visibility( TEveElement * tracker, bool visibility )
{
  std::vector<std::string> endcap;
  endcap.push_back("tec:TEC");
  endcap.push_back("tidf:TIDF");
  endcap.push_back("tidb:TIDB");
  endcap.push_back("pixfwd:PixelForwardZPlus");
  endcap.push_back("pixfwd:PixelForwardZMinus");
  set_children_visibility( tracker, "tracker:Tracker", endcap, visibility );
}

// show Tracker's Endcaps
void show_tracker_endcap( TEveElement * tracker )
{
  set_tracker_endcap_visibility( tracker, true );
}

// hide Tracker's Endcaps
void hide_tracker_endcap( TEveElement * tracker )
{
  set_tracker_endcap_visibility( tracker, false );
}

