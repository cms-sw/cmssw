#ifndef ExampleClass_H
#define ExampleClass_H

/** \class ExampleClass
 *  An example of doxygen-documented class conforming to the CMS style rules.
 *
 *  Features:<br>
 *  -doxygen-style header (note the \class directive)<br>
 *  -doxygen-like member function documentation<br>
 *  -Few setters and getters
 *
 *  $Date: 2005/07/26 10:13:49 $
 *  $Revision: 1.1 $
 *  \author W. Woodpecker - CERN
 */

#include <vector>

class SomeAlgorithm;


class ExampleClass {
 public:
  /// Constructor
  ExampleClass();

  /// Virtual Destructor
  virtual ~ExampleClass();

  /// A simple setter
  void setCount(int ticks);

  /// A simple getter
  int count() const;

  /// Another setter 
  void setValues(const std::vector<float>& entries);

  /// A getter returning a const reference
  const std::vector<float>& values() const;

  /// A member function
  float computeMean() const;

 protected:  

 private:
  int   theCount;          //< An int data member 
  std::vector<float> theValues; //< A vector data member
  SomeAlgorithm * theAlgo; //< A pointer data member

};
#endif // ExampleClass_H
