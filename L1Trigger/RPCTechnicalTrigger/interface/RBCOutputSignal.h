#ifndef RBCEmulator_RBCOutputSignal_h
#define RBCEmulator_RBCOutputSignal_h

class RBCId;
class RBCOutputSignal{
 public:
  RBCOutputSignal(const RBCId& id, int bx);
  virtual ~RBCOutputSignal();
  const RBCId& rbcid() const;
  int bx() const;
  bool operator < (const RBCOutputSignal& o) const;
 private:
  const RBCId& id;
  int  x;

};

#endif
