#ifndef eve_filter_h
#define eve_filter_h

#include <vector>
#include <string>

#include "TPRegexp.h"

class TEveElement;

// actions for apply_filter(...) and node_filter(...)
enum {
  do_nothing    = 0,
  do_hide       = 1,
  do_remove     = 2
};

// split a path into the name and a list of parent folders
void split_path(const std::string & path, std::string & name, std::vector<std::string> & parents);

// generalizes a node name of the form namespace:Name_NUM into a (regexp) filter that allows any number
TPRegexp make_filter(const std::string & token);

// generalizes a list of node names of the form namespace:Name_NUM into a (regexp) filter that allows any of the "namespace:Name" followd by any number
TPRegexp make_filter(const std::vector<std::string> & tokens);

// apply the filters to a node, specifying if the removed elements should be hidden (default), deleted (does not work) or ignored (i.e. do nothing)
void node_filter(TEveElement * node, int simplify = do_hide, bool verbose = false);

// initializes the filter static variables from a list of elements to display and colors to use
void init_filter(const std::vector< std::pair< std::string, Color_t> > & elements);

// dump the filters
void dump(void);

// apply the filters to a node, then notify it to update itself and its children for redrawing
void apply_filter(TEveElement * node, int simplify = do_hide, bool verbose = false);

// find a TEve root object (could be a TEveGeoRootNode or a TEveGeoShape) by its name
TEveElement * get_root_object(const char* name);

#endif // eve_filter_h
