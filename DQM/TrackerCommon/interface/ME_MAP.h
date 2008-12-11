#ifndef _ME_MAP_h
#define _ME_MAP_h

#include "DQMServices/Core/interface/MonitorElement.h"
#include <TCanvas.h>


typedef std::pair<std::string, MonitorElement *> entry;
typedef std::map<std::string, MonitorElement *> me_map;

class ME_MAP
{
 protected:

  /// a map of pointers to a subset of the MEs in contents:
  me_map mymap;

 public:
  
  ME_MAP() 
    { 
      mymap.clear();
    }

  ~ME_MAP(){}

  me_map get_me_map() const
    {
      return mymap;
    }

  int operator==(const ME_MAP &other) const
    {
      return (get_me_map() == other.get_me_map());
    }

  /// add the ME named "name" to the map
  void add(std::string name, MonitorElement *me_p);

  /// remove the ME named "name" from the map
  ///  void remove(std::string name);
  
  /// print the map into a gif named "name"
  void print(std::string name);

  /// clean old eps and gif files by the same name
  void clean_old(std::string gif_name);

  /// divide the canvas according to the number of elements to display
  void divide_canvas(int num_elements, TCanvas &canvas);

  /// create the gif file
  void create_gif(std::string name);

  void clear()
    {
      mymap.clear();
    }
};

#endif
