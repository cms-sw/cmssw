#ifndef PhysicsTools_NanoAOD_RNTupleFieldPtr_h
#define PhysicsTools_NanoAOD_RNTupleFieldPtr_h

#include <ROOT/RNTupleModel.hxx>
#include <ROOT/REntry.hxx>

template <typename T>
class RNTupleFieldPtr {
public:
  RNTupleFieldPtr() = default;
  explicit RNTupleFieldPtr(const std::string& name, const std::string& desc, ROOT::RNTupleModel& model) : m_name(name) {
    m_field = model.MakeField<T>(m_name, desc);
    // A bare model has no default entry to allocate the value in, and MakeField hands back nothing,
    // so the value lives here instead and reaches an entry through bind(). The Events model is bare
    // because ROOT only lets an untyped record be extended with new subfields (the trigger backfill)
    // in a bare model; the Runs, LuminosityBlocks and provenance models keep their default entry.
    if (!m_field) {
      m_field = std::make_shared<T>();
    }
  }
  // Point an entry at this field's value. Redundant for a model with its own default entry, and
  // required for the bare Events model, once at the start and again after every schema update.
  void bind(ROOT::REntry& entry) const { entry.BindValue(m_name, m_field); }
  void fill(const T& value) { *m_field = value; }
  const std::string& getFieldName() const { return m_name; }

private:
  std::string m_name;
  std::shared_ptr<T> m_field;
};

#endif
