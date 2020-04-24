#ifndef RCTInfo_hh
#define RCTInfo_hh

class RCTInfo {
 public:
  RCTInfo() {
    crateID = 0;
    linkIDEven = 0;
    linkIDOdd  = 0;
    c1BC0 = 0;
    c2BC0 = 0;
    c3BC0 = 0;
    c4BC0 = 0;
    c5BC0 = 0;
    c6BC0 = 0;
    for(int i = 0; i < 4; i++) ieRank[i] = 0;
    for(int i = 0; i < 4; i++) ieCard[i] = 0;
    for(int i = 0; i < 4; i++) ieRegn[i] = 0;
    mBits = 0;
    qBits = 0;
    for(int i = 0; i < 4; i++) neRank[i] = 0;
    for(int i = 0; i < 4; i++) neCard[i] = 0;
    for(int i = 0; i < 4; i++) neRegn[i] = 0;
    oBits = 0;
    tBits = 0;
    for(int i = 0; i < 2; i++) {
      for(int j = 0; j < 4; j++) {
	hfEt[i][j] = 0;
      }
    }
    for(int i = 0; i < 7; i++) {
      for(int j = 0; j < 2; j++) {
	rgnEt[i][j] = 0;
      }
    }

    hfQBits = 0;
  }
  RCTInfo(const RCTInfo& in) {
    crateID = in.crateID;
    linkIDEven = in.linkIDEven;
    linkIDOdd  = in.linkIDOdd;
    c1BC0 = in.c1BC0;
    c2BC0 = in.c2BC0;
    c3BC0 = in.c3BC0;
    c4BC0 = in.c4BC0;
    c5BC0 = in.c5BC0;
    c6BC0 = in.c6BC0;
    for(int i = 0; i < 4; i++) ieRank[i] = in.ieRank[i];
    for(int i = 0; i < 4; i++) ieCard[i] = in.ieCard[i];
    for(int i = 0; i < 4; i++) ieRegn[i] = in.ieRegn[i];
    mBits = in.mBits;
    qBits = in.qBits;
    for(int i = 0; i < 4; i++) neRank[i] = in.neRank[i];
    for(int i = 0; i < 4; i++) neCard[i] = in.neCard[i];
    for(int i = 0; i < 4; i++) neRegn[i] = in.neRegn[i];
    oBits = in.oBits;
    tBits = in.tBits;
    for(int i = 0; i < 2; i++) {
      for(int j = 0; j < 4; j++) {
	hfEt[i][j] = in.hfEt[i][j];
      }
    }
    for(int i = 0; i < 7; i++) {
      for(int j = 0; j < 2; j++) {
	rgnEt[i][j] = in.rgnEt[i][j];
      }
    }
    hfQBits = in.hfQBits;
  }
  void operator=(const RCTInfo& in) {
    this->crateID = in.crateID;
    this->linkIDEven = in.linkIDEven;
    this->linkIDOdd  = in.linkIDOdd;
    this->c1BC0 = in.c1BC0;
    this->c2BC0 = in.c2BC0;
    this->c3BC0 = in.c3BC0;
    this->c4BC0 = in.c4BC0;
    this->c5BC0 = in.c5BC0;
    this->c6BC0 = in.c6BC0;
    for(int i = 0; i < 4; i++) this->ieRank[i] = in.ieRank[i];
    for(int i = 0; i < 4; i++) this->ieCard[i] = in.ieCard[i];
    for(int i = 0; i < 4; i++) this->ieRegn[i] = in.ieRegn[i];
    this->mBits = in.mBits;
    this->qBits = in.qBits;
    for(int i = 0; i < 4; i++) this->neRank[i] = in.neRank[i];
    for(int i = 0; i < 4; i++) this->neCard[i] = in.neCard[i];
    for(int i = 0; i < 4; i++) this->neRegn[i] = in.neRegn[i];
    this->oBits = in.oBits;
    this->tBits = in.tBits;
    for(int i = 0; i < 2; i++) {
      for(int j = 0; j < 4; j++) {
	this->hfEt[i][j] = in.hfEt[i][j];
      }
    }
    for(int i = 0; i < 7; i++) {
      for(int j = 0; j < 2; j++) {
	this->rgnEt[i][j] = in.rgnEt[i][j];
      }
    }
    this->hfQBits = in.hfQBits;
  }
  unsigned int crateID;
  unsigned int linkIDEven;
  unsigned int linkIDOdd; 
  unsigned int c1BC0;
  unsigned int c2BC0;
  unsigned int c3BC0;
  unsigned int c4BC0;
  unsigned int c5BC0;
  unsigned int c6BC0;
  unsigned int ieRank[4];
  unsigned int ieCard[4];
  unsigned int ieRegn[4];
  unsigned int mBits;
  unsigned int qBits;
  unsigned int neRank[4];
  unsigned int neCard[4];
  unsigned int neRegn[4];    
  unsigned int oBits;
  unsigned int tBits;
  unsigned int hfEt[2][4];
  unsigned int rgnEt[7][2];
  unsigned int hfQBits;
};

#endif
