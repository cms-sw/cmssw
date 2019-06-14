#ifndef EventFilter_RPCRawToDigi_RPCAMCLinkEvent_h
#define EventFilter_RPCRawToDigi_RPCAMCLinkEvent_h

class RPCAMCLinkEvent {
public:
  static unsigned int const group_mask_ = 0x7000;
  static unsigned int const level_mask_ = 0x0700;
  static unsigned int const event_mask_ = 0x00ff;

  static unsigned int const not_set_ = 0x0000;

  static unsigned int const fed_ = 0x1000;
  static unsigned int const amc_ = 0x2000;
  static unsigned int const input_ = 0x3000;

  static unsigned int const debug_ = 0x0100;
  static unsigned int const info_ = 0x0200;
  static unsigned int const warn_ = 0x0300;
  static unsigned int const error_ = 0x0400;
  static unsigned int const fatal_ = 0x0500;

public:
  static unsigned int getGroup(unsigned int id);
  static unsigned int getLevel(unsigned int id);
  static unsigned int getEvent(unsigned int id);

  static unsigned int getId(unsigned int event, unsigned int group = not_set_, unsigned int level = not_set_);
};

inline unsigned int RPCAMCLinkEvent::getGroup(unsigned int id) { return (id & group_mask_); }

inline unsigned int RPCAMCLinkEvent::getLevel(unsigned int id) { return (id & level_mask_); }

inline unsigned int RPCAMCLinkEvent::getEvent(unsigned int id) { return (id & event_mask_); }

inline unsigned int RPCAMCLinkEvent::getId(unsigned int event, unsigned int group, unsigned int level) {
  return ((event & event_mask_) | (group & group_mask_) | (level & level_mask_));
}

#endif  // EventFilter_RPCRawToDigi_RPCAMCLinkEvent_h
