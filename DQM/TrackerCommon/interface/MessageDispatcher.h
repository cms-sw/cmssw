#ifndef _DQM_TrackerCommon_MessageDispatcher_h_
#define _DQM_TrackerCommon_MessageDispatcher_h_

#include <vector>
#include <string>

#include "xgi/Utils.h"
#include "xgi/Method.h"


enum MessageType { info = 0, warning = 1, error = 2 };


class Message
{

 private:

  std::string title;
  std::string text;
  MessageType type;

 public:

  Message(std::string the_title, std::string the_text, MessageType the_type)
    {
      type = the_type;
      title = the_title;
      text = the_text;
    }

  std::string getTitle() { return title; }
  std::string getText() { return text; }
  std::string getType();
};


class MessageDispatcher
{

 private:

  std::vector<Message *> undispatched;

 public:

  MessageDispatcher()
    {
    }

  void add(Message *new_message)
    {
      undispatched.push_back(new_message);
    }

  bool hasAnyMessages()
    { 
      return (!undispatched.empty()); 
    }

  void dispatchMessages(xgi::Output *out);

};

#endif
