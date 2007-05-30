#ifndef CondFormatsRPCObjectsLinkBoardPackedStrip_H
#define CondFormatsRPCObjectsLinkBoardPackedStrip_H

class LinkBoardPackedStrip {
public:

  LinkBoardPackedStrip(int packedStripLB) 
    : thePackedStrip(packedStripLB) { }

  LinkBoardPackedStrip(int febInLB, int stripPinInFeb) 
    : thePackedStrip( (febInLB-1)*16+stripPinInFeb-1) { } 

  int febInLB() const { return thePackedStrip/16+1; } 
  int stripPinInFeb() const { return thePackedStrip%16+1; }
  int packedStrip() const { return thePackedStrip; }

private:
  int thePackedStrip; 
};
#endif
