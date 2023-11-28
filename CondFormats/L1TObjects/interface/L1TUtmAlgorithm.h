//
// NOTE:  This file was automatically generated from UTM library via import_utm.pl
// DIRECT EDITS MIGHT BE LOST.
//
/**
 * @author      Takashi Matsushita
 * Created:     12 Mar 2015
 */

#ifndef tmEventSetup_L1TUtmAlgorithm_hh
#define tmEventSetup_L1TUtmAlgorithm_hh

#include <set>
#include <string>
#include <vector>
#include "CondFormats/Serialization/interface/Serializable.h"
#include "tmEventSetup/esAlgorithm.hh"

/**
 *  This class implements data structure for Algorithm
 */
class L1TUtmAlgorithm {
public:
  L1TUtmAlgorithm()
      : name_(),
        expression_(),
        expression_in_condition_(),
        rpn_vector_(),
        index_(),
        module_id_(),
        module_index_(),
        version(0){};
  L1TUtmAlgorithm(std::string name,
                  std::string expression,
                  std::string expression_in_condition,
                  std::vector<std::string> rpn_vector,
                  unsigned int index,
                  unsigned int module_id,
                  unsigned int module_index,
                  unsigned int ver)
      : name_(name),
        expression_(expression),
        expression_in_condition_(expression_in_condition),
        rpn_vector_(rpn_vector),
        index_(index),
        module_id_(module_id),
        module_index_(module_index),
        version(ver){};

  L1TUtmAlgorithm(const tmeventsetup::esAlgorithm& esAlg)
      : L1TUtmAlgorithm(esAlg.getName(),
                        esAlg.getExpression(),
                        esAlg.getExpressionInCondition(),
                        esAlg.getRpnVector(),
                        esAlg.getIndex(),
                        esAlg.getModuleId(),
                        esAlg.getModuleIndex(),
                        0  //There is no version retrieval in esAlgorithm. However, it seems pretty hard coded to 0
        ){};

  virtual ~L1TUtmAlgorithm() = default;

  /** set rpn_vector_ */
  void setRpnVector(const std::vector<std::string>& x) { rpn_vector_ = x; };

  /** get algorithm name */
  const std::string& getName() const { return name_; };

  /** get algorithm expression in grammar */
  const std::string& getExpression() const { return expression_; };

  /** get algorithm expression in condition */
  const std::string& getExpressionInCondition() const { return expression_in_condition_; };

  /** get reverse polish notion of algorithm expression in condition */
  const std::vector<std::string>& getRpnVector() const { return rpn_vector_; };

  /** get algorithm index */
  unsigned int getIndex() const { return index_; };

  /** get module id */
  unsigned int getModuleId() const { return module_id_; };

  /** get module index */
  unsigned int getModuleIndex() const { return module_index_; };

protected:
  std::string name_;                    /**< name of algorithm */
  std::string expression_;              /**< algorithm expression in grammar */
  std::string expression_in_condition_; /**< algorithm expression in condition */
  std::vector<std::string> rpn_vector_; /**< reverse polish notation of algorithm expression in condition */
  unsigned int index_;                  /**< index of algorithm (global) */
  unsigned int module_id_;              /**< module id */
  unsigned int module_index_;           /**< index of algorithm in module (local to module id) */
  unsigned int version;

  COND_SERIALIZABLE;
};

#endif  // tmEventSetup_L1TUtmAlgorithm_hh
