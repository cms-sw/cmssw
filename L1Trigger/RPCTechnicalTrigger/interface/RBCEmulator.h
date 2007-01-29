#ifndef RBCEmulator_RBCEmulator_h
#define RBCEmulator_RBCEmulator_h
class RBCLogic;
class RBCPolicy;
class RBCOutputSignalContainer;
class RBCEmulator{
 public:
  RBCEmulator();
  virtual ~RBCEmulator();
  void emulate(RBCPolicy* policy);
  RBCOutputSignalContainer triggers();
 private:
  RBCLogic* l;
};
#endif