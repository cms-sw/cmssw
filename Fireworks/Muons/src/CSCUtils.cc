#include "Fireworks/Muons/interface/CSCUtils.h"
#include <iostream>

namespace fireworks
{
  void fillCSCChamberParameters(int station, int ring, 
                                double& length, double& thickness)
  {
    thickness = 18.0;

    if ( ring == 3 )
    {
      assert(station == 1); // Only station 1 has a 3rd ring
      length = 179.3;
      return;
    }

    else if ( ring == 1 )
    {
      if ( station == 1 )
      {
        length = 162.0;
        thickness = 14.7; // ME1/1
        return;
      }
      else if ( station == 2 )
      {
        length = 204.6;
        return;
      }
      else if ( station == 3 )
      {
        length = 184.6; 
        return;
      }
      else if ( station == 4 )
      {
        length = 166.7;
        return;
      }
      else
        return;
    }
  
    else if ( ring == 2 )
    {
      if ( station == 2 || station == 3 || station == 4 )
      {
        length = 338.0;
        return;
      }
      else if ( station == 1 )
      {
        length = 189.4;
        return;
      }
      else 
        return;
    }
    
    else
      return;
  }
}       

