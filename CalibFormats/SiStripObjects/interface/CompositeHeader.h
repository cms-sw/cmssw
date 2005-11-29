#ifndef CALIBTRACKER_SISTRIPCONNECTIVITY_COMPOSITEHEADER_H
#define CALIBTRACKER_SISTRIPCONNECTIVITY_COMPOSITEHEADER_H
#include <string>

/**
 * completely empty, used to add header infos
 */

class CompositeHeader {
 public:
  
  void setComment(string in) {comment = in;}
  string getComment() {return comment;}
 
 private:
  string   comment;

};

#endif
