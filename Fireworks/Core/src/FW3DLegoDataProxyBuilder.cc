// -*- C++ -*-
//
// Package:     Core
// Class  :     FW3DLegoDataProxyBuilder
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Thu Dec  6 17:49:54 PST 2007
// $Id: FW3DLegoDataProxyBuilder.cc,v 1.1 2007/12/09 22:49:23 chrjones Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FW3DLegoDataProxyBuilder.h"


//
// constants, enums and typedefs
//
namespace fw3dlego
{
  const double xbins[79] = {
	   -4.716, -4.538, -4.363, -4.191, -4.013, -3.839, -3.664, -3.489, -3.314, 
	   -3.139, -2.964, -2.853, -2.650, -2.500, -2.322, -2.172, -2.043, -1.930, -1.830, 
	   -1.740, -1.653, -1.566, -1.479, -1.392, -1.305, -1.218, -1.131, -1.044, -0.957, 
	   -0.870, -0.783, -0.696, -0.609, -0.522, -0.435, -0.348, -0.261, -0.174, -0.087,  
	   0.000,
	    0.087,  0.174,  0.261,  0.348,  0.435,  0.522,  0.609,  0.696,  0.783,  0.870,
	    0.957,  1.044,  1.131,  1.218,  1.305,  1.392,  1.479,  1.566,  1.653,  1.740,  
	    1.830,  1.930,  2.043,  2.172,  2.322,  2.500,  2.650,  2.853,  2.964,  3.139,  
	    3.314,  3.489,  3.664,  3.839,  4.013,  4.191,  4.363,  4.538,  4.716};

}

//
// static data member definitions
//

//
// constructors and destructor
//
FW3DLegoDataProxyBuilder::FW3DLegoDataProxyBuilder():
  m_item(0)
{
}

// FW3DLegoDataProxyBuilder::FW3DLegoDataProxyBuilder(const FW3DLegoDataProxyBuilder& rhs)
// {
//    // do actual copying here;
// }

FW3DLegoDataProxyBuilder::~FW3DLegoDataProxyBuilder()
{
}

//
// assignment operators
//
// const FW3DLegoDataProxyBuilder& FW3DLegoDataProxyBuilder::operator=(const FW3DLegoDataProxyBuilder& rhs)
// {
//   //An exception safe implementation is
//   FW3DLegoDataProxyBuilder temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void
FW3DLegoDataProxyBuilder::setItem(const FWEventItem* iItem)
{
  m_item = iItem;
}

void
FW3DLegoDataProxyBuilder::build(TH2F** iObject)
{
  if(0!= m_item) {
    build(m_item, iObject);
  }
}
//
// const member functions
//

//
// static member functions
//
