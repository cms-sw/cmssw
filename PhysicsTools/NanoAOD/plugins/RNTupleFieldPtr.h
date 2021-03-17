#ifndef PhysicsTools_NanoAOD_RNTupleFieldPtr_h
#define PhysicsTools_NanoAOD_RNTupleFieldPtr_h

#include <ROOT/RNTupleModel.hxx>
using ROOT::Experimental::RNTupleModel;

template <typename T>
class RNTupleFieldPtr {
public:
  RNTupleFieldPtr() = default;
  explicit RNTupleFieldPtr(const std::string& name, RNTupleModel& model)
      : m_name(name)
  {
    m_field = model.MakeField<T>(m_name);
  }
  void fill(const T& value) {
    *m_field = value;
  }
  const std::string& getFieldName() const {
    return m_name;
  }
private:
  std::string m_name;
  std::shared_ptr<T> m_field;
};

#endif
