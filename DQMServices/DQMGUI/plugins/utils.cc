// $Id: utils.cc,v 1.6 2009/12/15 22:56:08 ameyer Exp $

/*!
  \file utils.cc
  \version $Revision: 1.6 $
  \date $Date: 2009/12/15 22:56:08 $
*/

#include "utils.h"

#include "TH2.h"
#include "TStyle.h"
#include "TColor.h"
#include "TROOT.h"

using namespace dqm;
bool utils::init = false;
void utils::reportSummaryMapPalette(TH2* obj)
{
  static int pcol[20];

  if( ! utils::init )
  {
    utils::init = true;

    float rgb[20][3];

    for( int i=0; i<20; i++ )
    {
      if ( i < 17 )
      {
        rgb[i][0] = 0.80+0.01*i;
        rgb[i][1] = 0.00+0.03*i;
        rgb[i][2] = 0.00;
      }
      else if ( i < 19 )
      {
        rgb[i][0] = 0.80+0.01*i;
        rgb[i][1] = 0.00+0.03*i+0.15+0.10*(i-17);
        rgb[i][2] = 0.00;
      }
      else if ( i == 19 )
      {
        rgb[i][0] = 0.00;
        rgb[i][1] = 0.80;
        rgb[i][2] = 0.00;
      }
      pcol[i] = TColor::GetColor(rgb[i][0], rgb[i][1], rgb[i][2]);
    }
  }

  gStyle->SetPalette(20, pcol);

  if( obj )
  {
    obj->SetMinimum(-1.e-15);
    obj->SetMaximum(+1.0);
    obj->SetOption("colz");
  }
}
