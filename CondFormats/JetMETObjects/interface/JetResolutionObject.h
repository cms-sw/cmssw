#ifndef JetResolutionObject_h
#define JetResolutionObject_h

// If you want to use the JER code in standalone mode, you'll need to create a new define named 'STANDALONE'. If you use gcc for compiling, you'll need to add
// -DSTANDALONE to the command line
// In standalone mode, no reference to CMSSW exists, so the only way to retrieve resolutions and scale factors are from text files.

#ifndef STANDALONE
#include "CondFormats/Serialization/interface/Serializable.h"
#else
// Create no-op definitions of CMSSW macro
#define COND_SERIALIZABLE
#define COND_TRANSIENT
#endif

#include <unordered_map>
#include <vector>
#include <string>
#include <tuple>
#include <memory>
#include <initializer_list>

#ifndef STANDALONE
#include "CommonTools/Utils/interface/FormulaEvaluator.h"
#else
#include <TFormula.h>
#endif

enum class Variation { NOMINAL = 0, DOWN = 1, UP = 2 };

template <typename T>
T clip(const T& n, const T& lower, const T& upper) {
  return std::max(lower, std::min(n, upper));
}

namespace JME {
  template <typename T, typename U>
  struct bimap {
    typedef std::unordered_map<T, U> left_type;
    typedef std::unordered_map<U, T> right_type;

    left_type left;
    right_type right;

    bimap(std::initializer_list<typename left_type::value_type> l) {
      for (auto& v : l) {
        left.insert(v);
        right.insert(typename right_type::value_type(v.second, v.first));
      }
    }

    bimap() {
      // Empty
    }

    bimap(bimap&& rhs) {
      left = std::move(rhs.left);
      right = std::move(rhs.right);
    }
  };

  enum class Binning {
    JetPt = 0,
    JetEta,
    JetAbsEta,
    JetE,
    JetArea,
    Mu,
    Rho,
    NPV,
  };

};  // namespace JME

// Hash function for Binning enum class
namespace std {
  template <>
  struct hash<JME::Binning> {
    typedef JME::Binning argument_type;
    typedef std::size_t result_type;

    hash<uint8_t> int_hash;

    result_type operator()(argument_type const& s) const { return int_hash(static_cast<uint8_t>(s)); }
  };
};  // namespace std

namespace JME {

  class JetParameters {
  public:
    typedef std::unordered_map<Binning, float> value_type;

    JetParameters() = default;
    JetParameters(JetParameters&& rhs);
    JetParameters(std::initializer_list<typename value_type::value_type> init);

    JetParameters& setJetPt(float pt);
    JetParameters& setJetEta(float eta);
    JetParameters& setJetE(float e);
    JetParameters& setJetArea(float area);
    JetParameters& setMu(float mu);
    JetParameters& setRho(float rho);
    JetParameters& setNPV(float npv);
    JetParameters& set(const Binning& bin, float value);
    JetParameters& set(const typename value_type::value_type& value);

    static const bimap<Binning, std::string> binning_to_string;

    std::vector<float> createVector(const std::vector<Binning>& binning) const;
    std::vector<float> createVector(const std::vector<std::string>& binname) const;

  private:
    value_type m_values;
  };

  class JetResolutionObject {
  public:
    struct Range {
      float min;
      float max;

      Range() {
        // Empty
      }

      Range(float min, float max) {
        this->min = min;
        this->max = max;
      }

      bool is_inside(float value) const { return (value >= min) && (value < max); }

      COND_SERIALIZABLE;
    };

    class Definition {
    public:
      Definition() {
        // Empty
      }

      Definition(const std::string& definition);

      const std::vector<std::string>& getBinsName() const { return m_bins_name; }

      const std::vector<Binning>& getBins() const { return m_bins; }

      std::string getBinName(size_t bin) const { return m_bins_name[bin]; }

      size_t nBins() const { return m_bins_name.size(); }

      const std::vector<std::string>& getVariablesName() const { return m_variables_name; }

      const std::vector<Binning>& getVariables() const { return m_variables; }

      std::string getVariableName(size_t variable) const { return m_variables_name[variable]; }

      size_t nVariables() const { return m_variables_name.size(); }

      const std::vector<std::string>& getParametersName() const { return m_parameters_name; }

      size_t nParameters() const { return m_parameters_name.size(); }

      std::string getFormulaString() const { return m_formula_str; }

#ifndef STANDALONE
      const reco::FormulaEvaluator* getFormula() const { return m_formula.get(); }
#else
      TFormula const* getFormula() const { return m_formula.get(); }
#endif
      void init();

    private:
      std::vector<std::string> m_bins_name;
      std::vector<std::string> m_variables_name;
      std::string m_formula_str;

#ifndef STANDALONE
      std::shared_ptr<reco::FormulaEvaluator> m_formula COND_TRANSIENT;
#else
      std::shared_ptr<TFormula> m_formula COND_TRANSIENT;
#endif
      std::vector<Binning> m_bins COND_TRANSIENT;
      std::vector<Binning> m_variables COND_TRANSIENT;
      std::vector<std::string> m_parameters_name COND_TRANSIENT;

      COND_SERIALIZABLE;
    };

    class Record {
    public:
      Record() {
        // Empty
      }

      Record(const std::string& record, const Definition& def);

      const std::vector<Range>& getBinsRange() const { return m_bins_range; }

      const std::vector<Range>& getVariablesRange() const { return m_variables_range; }

      const std::vector<float>& getParametersValues() const { return m_parameters_values; }

      size_t nVariables() const { return m_variables_range.size(); }

      size_t nParameters() const { return m_parameters_values.size(); }

    private:
      std::vector<Range> m_bins_range;
      std::vector<Range> m_variables_range;
      std::vector<float> m_parameters_values;

      COND_SERIALIZABLE;
    };

  public:
    JetResolutionObject(const std::string& filename);
    JetResolutionObject(const JetResolutionObject& filename);
    JetResolutionObject();

    void dump() const;
    void saveToFile(const std::string& file) const;

    const Record* getRecord(const JetParameters& bins) const;
    float evaluateFormula(const Record& record, const JetParameters& variables) const;

    const std::vector<Record>& getRecords() const { return m_records; }

    const Definition& getDefinition() const { return m_definition; }

  private:
    Definition m_definition;
    std::vector<Record> m_records;

    bool m_valid = false;

    COND_SERIALIZABLE;
  };
};  // namespace JME

#endif
