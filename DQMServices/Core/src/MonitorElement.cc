#include "DQMServices/Core/interface/MonitorElement.h"

#include <iostream>

using namespace std;

MonitorElement::MonitorElement() 
{
  softReset_on = accumulate_on = false;
}

MonitorElement::MonitorElement(const char*name) 
{
  softReset_on = accumulate_on = false;
}

MonitorElement::~MonitorElement() {}

bool MonitorElement::wasUpdated() const {return man.variedSince;}

bool MonitorElement::isUrgent() const {return man.urgent;}

void MonitorElement::update() {man.variedSince = true;}

void MonitorElement::setUrgent() 
{
  cout << " Status for monitoring element " << getName() << " set to urgent" 
       << endl;
  man.urgent = true;
}

void MonitorElement::resetUpdate() {man.variedSince = false;}

bool MonitorElement::isFolder(void) const {return man.folder_flag;}
bool MonitorElement::isNotFolder(void) const {return !isFolder();}

// true if ME should be reset at end of monitoring cycle
bool MonitorElement::resetMe(void) const{return man.resetMe;}

// set resetMe flag (default: false)
void MonitorElement::setResetMe(bool flag)
{
  cout << " \"resetMe\" flag for monitoring element " << getName() 
       << " set to";
  if(flag)
    cout << " true";
  else
    cout << " false";
  cout << endl;

  man.resetMe = flag;
}

// if true, will accumulate ME contents (over many periods)
// until method is called with flag = false again
void MonitorElement::setAccumulate(bool flag)
{
  accumulate_on = flag;

  cout << " \"accumulate\" option has been";
  if(accumulate_on)
    cout << " en";
  else
    cout << " dis";
  cout << "abled for " << getName() << endl;  
}
