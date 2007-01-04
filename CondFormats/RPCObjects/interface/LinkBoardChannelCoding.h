#ifndef CondFormatsRPCObjectsLinkBoardChannelCoding_H
#define CondFormatsRPCObjectsLinkBoardChannelCoding_H

class LinkBoardChannelCoding {
public:

  LinkBoardChannelCoding(int channelLB) 
    : theChannel(channelLB) { }

  LinkBoardChannelCoding(int fedInLB, int stripPinInFeb) 
    : theChannel( (fedInLB-1)*16+stripPinInFeb-1) { } 

  int fedInLB() const { return theChannel/16+1; } 
  int stripPinInFeb() const { return theChannel%16+1; }
  int channel() const { return theChannel; }

private:
  int theChannel; 
};
#endif
