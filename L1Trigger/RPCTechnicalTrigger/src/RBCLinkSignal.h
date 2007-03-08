#ifndef RPCTechnicalTrigger_RBCLinkSignal_h
#define RPCTechnicalTrigger_RBCLinkSignal_h
class RBCId;
class RBCLinkSignal {
 public:
  RBCLinkSignal(const RBCId & id, int layer, int bx);
  virtual ~RBCLinkSignal();
  const RBCId& rbcid() const;
  int triggerLayer() const;
  int bx() const;
  bool operator < (const RBCLinkSignal& link) const;
 private:
  const RBCId& rid;
  int l;
  int x;

};
#endif
