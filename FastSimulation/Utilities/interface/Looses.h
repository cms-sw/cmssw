#ifndef LOOSES_H
#define LOOSES_H

//C++ headers
#include <string>
#include <map>
#include <vector>

class Looses {
public:
  ///Constructor is not public (only one instance needed)
  static Looses* instance();

  ///Virtual destructor (empty)
  virtual ~Looses();

  ///Counting
  void count(const std::string& name, unsigned cut);

  ///Printing
  void summary();

 private:
  // The constructor is hidden as we do not want to construct
  // more than one instance.
  Looses();

  // The instance
  static Looses* myself;

  // The table of losses
  std::map< std::string, std::vector<unsigned> > theLosses;
};
#endif
