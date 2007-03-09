#ifndef HTrackAssociator_HTimerStack_h
#define HTrackAssociator_HTimerStack_h 1
#include "Utilities/Timing/interface/TimingReport.h"
#include <stack>
class HTimerStack
{
 public:
   ~HTimerStack()
     {
	clean_stack();
     }
   
   void push(std::string name){
      if( (*TimingReport::current())["firstcall_"+name].counter == 0)
	stack.push(new TimeMe("firstcall_"+name));
      else
	stack.push(new TimeMe(name));
   }
   
   void pop(){
      if (!stack.empty()) {
	 delete stack.top();
	 stack.pop();
      }
   }

   void clean_stack(){
      while(!stack.empty()) pop();
   }
   
   void pop_and_push(std::string name) {
      pop();
      push(name);
   }
   
 private:
   std::stack<TimeMe*> stack;
};
#endif
