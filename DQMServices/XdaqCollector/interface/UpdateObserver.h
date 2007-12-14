#ifndef DQMSERVICES_COMPONENTS_UPDATEOBSERVER_H
#define DQMSERVICES_COMPONENTS_UPDATEOBSERVER_H

namespace dqm
{

  class UpdateObserver
  { 
  public:

    UpdateObserver(){}
    virtual ~UpdateObserver(){}
    virtual void onUpdate() const = 0;

  private:
    
    

  };
}
#endif
