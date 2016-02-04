#ifndef _flashlist_elements_h_
#define _flashlist_elements_h_

#include "xdata/Bag.h"
#include "xdata/String.h"
//#include "xdata/Boolean.h"
//#include "xdata/Integer.h"

namespace xmas2dqm 
{
	namespace wse 
	{
		
		class FlashlistElements
		{
		public:
			FlashlistElements();
			//~ServiceProperties();
			void registerFields(xdata::Bag<FlashlistElements> * bag);
			//static bool less(ServiceProperties s1, ServiceProperties s2);
			xdata::String flashlist;
			xdata::String element;
			xdata::String xtitle;
			xdata::String ytitle;
			xdata::String bins;
			xdata::String xmin;
			xdata::String xmax;
		};
	}
}

#endif
