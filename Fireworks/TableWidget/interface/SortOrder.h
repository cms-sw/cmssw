#ifndef Fireworks_TableWidget_SortOrder_h
#define Fireworks_TableWidget_SortOrder_h
// -*- C++ -*-
//
// Package:     TableWidget
// Class  :     SortOrder
// 
/**\class SortOrder SortOrder.h Fireworks/TableWidget/src/SortOrder.h

 Description: Enumeration used to describe the sort ordering of a table

 Usage:
    Used internally by the table widget to denote the order in which rows have been sorted

*/
//
// Original Author:  Chris Jones
//         Created:  Mon Feb  2 19:10:26 EST 2009
// $Id: SortOrder.h,v 1.1 2009/02/03 20:33:03 chrjones Exp $
//

// system include files

// user include files

// forward declarations

namespace fireworks{
   namespace table {
      enum SortOrder { kNotSorted, kAscendingSort, kDescendingSort};      
   }
}

#endif
