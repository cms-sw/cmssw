#ifndef eve_macros_h
#define eve_macros_h

#include <vector>
#include <string>
#include <Rtypes.h>
class TEveElement;

// get the name from an object derived from both TEveElement and TNamed
const char* get_name( const TEveElement * element );

// get the title from an object derived from both TEveElement and TNamed
const char* get_title( const TEveElement * element );

// force a node to expand its internal reprsentation, so all children are actually present
void expand_node( TEveElement * element );

// retrieve a TShape from a TEveElement
const TGeoShape * get_shape( const TEveElement * element );

// overloaded non-const TShape retrieval, allowed from a TGeoShape only
TGeoShape * get_shape( TEveElement * element );

// return a copy of the local-to-global transformation applied to a TEveElement
TGeoMatrix * get_transform( const TEveElement * element );

// clone a TEveGeoShape or TEveGeoNode into a new TEveGeoShape, and add it as a child to a parent if one is given
TEveGeoShape * clone( const TEveElement * element, TEveElement * parent = 0);

// set an element's color and alpha, and possibly its children's up to levels levels deep
void set_color( TEveElement * element, Color_t color, float alpha = 1.0, unsigned int levels = 0 );

// check if a node has any children or if it's a leaf node
bool is_leaf_node( const TEveElement * element );

// toggle an elements's children visibility, based on their name
// names are checked only up to their length, so for example tec:TEC will match both tec:TEC_1 and tec:TEC_2
void set_children_visibility( TEveElement * element, const std::string & node_name, const std::vector<std::string> & children_name, bool visibility );

// set Tracker's Endcaps visibility
void set_tracker_endcap_visibility( TEveElement * tracker, bool visibility );

// show Tracker's Endcaps
void show_tracker_endcap( TEveElement * tracker );

// hide Tracker's Endcaps
void hide_tracker_endcap( TEveElement * tracker );

#endif // eve_macros_h
