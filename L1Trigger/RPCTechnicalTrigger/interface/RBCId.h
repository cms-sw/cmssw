#ifndef RPCTechnicalTrigger_RBCId_h
#define RPCTechnicalTrigger_RBCId_h

class RBCId
{
 public:
  RBCId();
  RBCId(int wheel, int sector);
  virtual ~RBCId();
  int wheel() const;
  int sector() const;
 private:
  int w;
  int s;

};
#endif
