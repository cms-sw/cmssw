#ifndef Tracker_TkFiniteStateMachine_h
#define Tracker_TkFiniteStateMachine_h

#include <string>
/**
 * Implement a state machine. Each of the ApvAnalysis component can be
 * - calibrating: not yet able to give an answer
 * - ready: can give an answer, the calibration stays as it is.
 * - calibrating:  can give an answer, at the same time the calibration is updated.
 * - stuck: a serious error happened.
 */
class TkStateMachine{
 public:
  
  enum StatusType {ready=1,calibrating=2,updating=3,stuck=4};
  
  bool alreadyCalibrated() const {return (myStatus == updating || myStatus == ready);}
  StatusType status() const {return myStatus;}
  
  void setReady() {myStatus = ready;}
  void setUpdating() {myStatus = updating;}
  void setCalibrating() {myStatus = calibrating;}
  void setStuck() {myStatus = stuck;}
  
  void setStatus(StatusType in) {myStatus = in;}
  
  bool isReady() const {return myStatus==ready;}
  bool isStuck() const {return myStatus==stuck;}
  bool isUpdating() const {return myStatus==updating;}
  bool isCalibrating() const {return myStatus==calibrating;}
  
  
  std::string statusName() {
    if (myStatus == ready) return "Ready";
    if (myStatus == calibrating) return "Calibrating";
    if (myStatus == updating) return "Updating";
    if (myStatus == stuck) return "Stuck";
    return "Unknown Status";
  }
  

 public:

  StatusType myStatus;

};

#endif


