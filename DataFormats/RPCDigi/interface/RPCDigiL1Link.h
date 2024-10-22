#ifndef RPCOBJECTS_RPCDIGIL1LINK_H
#define RPCOBJECTS_RPCDIGIL1LINK_H

#include <vector>
#include <map>

class RPCDigiL1Link {
public:
  RPCDigiL1Link();

  ~RPCDigiL1Link();

  bool empty() const;

  // Getters -- layer runs from 1 to nlayer
  bool empty(unsigned int layer) const;
  unsigned int rawdetId(unsigned int layer) const;
  int strip(unsigned int layer) const;
  int bx(unsigned int layer) const;
  unsigned int nlayer() const;

  // Setters --layer run from 1 to nlayer
  void setLink(unsigned int layer, unsigned int rpcdetId, int strip, int bx);

private:
  void checklayer(unsigned int layer) const;

private:
  std::vector<std::pair<unsigned int, int> > _link;
};
#endif
