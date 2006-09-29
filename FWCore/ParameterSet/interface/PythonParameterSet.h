#ifndef PythonParameterSet_h
#define PythonParameterSet_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <boost/python.hpp>
#include "FWCore/ParameterSet/src/PythonWrapper.h"

class PythonParameterSet
{
public:
  PythonParameterSet();
/*
  void addInt32(const std::string & name, int value);
  void addUInt32(const std::string & name, unsigned int value);
  void addVInt32(const std::string & name, const std::vector<int> & value);
  void addVUInt32(const std::string & name, const std::vector<unsigned int> &  value);
  void addUntrackedInt32(const std::string & name, int value);
  void addUntrackedUInt32(const std::string & name, unsigned int value);
  void addUntrackedVInt32(const std::string & name, const std::vector<int> & value);
  void addUntrackedVUInt32(const std::string & name, const std::vector<unsigned int> &  value);

  void addInt64(const std::string & name, int value);
  void addUInt64(const std::string & name, unsigned int value);
  void addVInt64(const std::string & name, const std::vector<int> & value);
  void addVUInt64(const std::string & name, const std::vector<unsigned int> &  value);
  void addUntrackedInt64(const std::string & name, int value);
  void addUntrackedUInt64(const std::string & name, unsigned int value);
  void addUntrackedVInt64(const std::string & name, const std::vector<int> & value);
  void addUntrackedVUInt64(const std::string & name, const std::vector<unsigned int> &  value);

  void addDouble(const std::string & name, double value);
  void addVDouble(const std::string & name, const std::vector<double> & value);
  void addUntrackedDouble(const std::string & name, double value);
  void addUntrackedVDouble(const std::string & name, const std::vector<double> & value);

  void addString(const std::string & name, const std::string & value);
  void addVString(const std::string & name, const std::vector<std::string> & value);
  void addUntrackedString(const std::string & name, const std::string & value);
  void addUntrackedVString(const std::string & name, const std::vector<std::string> & value);
*/
    template <class T>
    T
    getParameter(bool tracked, std::string const& name) const 
    {
      T result;
      if(tracked)
      {
        result = theParameterSet.template getParameter<T>(name);
      }
      else 
      {
        result = theParameterSet.template getUntrackedParameter<T>(name, result);
      }
      return result;
    }


    template <class T>
    void
    addParameter(bool tracked, std::string const& name, T value)
    {
     if(tracked)
     {
       theParameterSet.template addParameter<T>(name, value);
     }
     else 
     {
       theParameterSet.template addUntrackedParameter<T>(name, value);
     }
    }


    /// templated on the type of the contained object
    template <class T>
    boost::python::list
    getParameters(bool tracked, const std::string & name) const
    {
      std::vector<T> v = getParameter<std::vector<T> >(tracked, name);
      return edm::toPythonList(v);
    }

    /// unfortunate side effect: destroys the original list!
    template <class T>
    void
    addParameters(bool tracked, std::string const& name, boost::python::list  value)
    {
      std::vector<T> v = edm::toVector<T>(value);
      addParameter(tracked, name, v);
    }

private:
  edm::ParameterSet theParameterSet;
};

#endif

