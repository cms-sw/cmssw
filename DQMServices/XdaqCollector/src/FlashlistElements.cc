#include "DQMServices/XdaqCollector/interface/FlashlistElements.h"

namespace xmas2dqm 
{
	namespace wse 
	{

		FlashlistElements::FlashlistElements()
			:
			flashlist("flashlist_const"),
			element("element_const"),
			xtitle("xtitle_const"),
			ytitle("ytitle_const")
		{
		}
		
		void FlashlistElements::registerFields(xdata::Bag<FlashlistElements> * bag) 
		{
			bag->addField("flashlist", &flashlist); 	       
			bag->addField("element", &element);
			bag->addField("xtitle", &xtitle); 
			bag->addField("ytitle", &ytitle);
			bag->addField("bins", &bins);
			bag->addField("xmin", &xmin);
			bag->addField("xmax", &xmax);
		}

		/*bool ServiceProperties::less(ServiceProperties s1, ServiceProperties s2)
		{ 
			return ( s1.priority_ <= s2.priority_ ); 
		}*/
	}
}
