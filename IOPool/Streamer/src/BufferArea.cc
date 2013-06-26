
#include "IOPool/Streamer/interface/EventBuffer.h"

#include <memory>
#include <list>

namespace edm {

namespace 
 {
   // --------------------------------------------
   // keep the buffers around since there are intended to be used
   // by multiple threads
   
   struct BufHolder
   {
     BufHolder() { }
     ~BufHolder()
     {
       while(!v_.empty())
         { EventBuffer* b = v_.front(); delete b; v_.pop_front(); }
     }
     std::list<EventBuffer*> v_;
   };
   BufHolder holder;
 }

  EventBuffer* getEventBuffer(int es, int qd)
  {
    std::auto_ptr<EventBuffer> b(new EventBuffer(es,qd));
    holder.v_.push_front(b.get());
    return b.release();
  }

}


