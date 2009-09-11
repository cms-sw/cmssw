#include "DQM/TrackerCommon/interface/ME_MAP.h"

#include <TColor.h>
#include <TVirtualPad.h>
#include <math.h>
#include <iostream>
#include <cstdlib>

void ME_MAP::add(std::string name, MonitorElement *me_p)
{
  entry newentry(name, me_p);
  mymap.insert(newentry);
}

/*
  void ME_MAP::remove(std::string name)
  { 
  MonitorElement *d_ptr = mymap[name];
   delete d_ptr;
   mymap.erase(name);
    }
*/
void ME_MAP::print(std::string name)
{
  std::string gif_name = name + ".gif";
      
  clean_old(gif_name);
      
  create_gif(gif_name);
}

void ME_MAP::clean_old(std::string gif_name)
{
  ::unlink(gif_name.c_str());
}

void ME_MAP::divide_canvas(int num_elements, TCanvas &canvas)
{
  if (num_elements < 2) 
    {
      canvas.Divide(1, 1);
      return;
    }
  if (num_elements == 2) 
    {
      canvas.Divide(2, 1);
      return;
    }
      
  int columns = static_cast<int>(sqrt(static_cast<float>(num_elements)));
  int rows = static_cast<int>(ceil(static_cast<float>(num_elements) / columns));
  canvas.Divide(columns, rows); 
}

void ME_MAP::create_gif(std::string name)
{ 
  int num_elements = mymap.size();
      
  // If we (still) don't have anything, create empty eps
  if (num_elements == 0) 
    {
      std::string command = "cp empty.eps " + name; 
      ::system(command.c_str());
     // std::cout << "ME_MAP has no elements" << std::endl;
      return;
    }
      
  else
    {
      
     // std::cout << "ME_MAP has " << mymap.size() << " elements" << std::endl;

      TCanvas canvas("display");
	        
      divide_canvas(num_elements, canvas);
      	  
      int i = 0;
      me_map::iterator it;
      for (it = mymap.begin(); it != mymap.end(); it++)
	{
	  // move to the next place in the canvas
	  TVirtualPad * current_pad = canvas.cd(i + 1);
	  Color_t color = TColor::GetColor("000000");
	  if (it->second->hasOtherReport()) color = TColor::GetColor ("#FCD116");
	  if (it->second->hasWarning()) color = TColor::GetColor ("#FF8000");
	  if (it->second->hasError()) color = TColor::GetColor ("#CC0000");
	  current_pad->HighLight(color, kTRUE);
	  it->second->getRootObject()->Draw();
	  i++;
	}
      	  
      canvas.SaveAs(name.c_str());
    }
}
  
