//
// NOTE:  This file was automatically generated from UTM library via import_utm.pl
// DIRECT EDITS MIGHT BE LOST.
//
/**
 * @author      Takashi Matsushita
 * Created:     8 Nov 2015
 */

/** @todo nope */

#ifndef tmEventSetup_L1TUtmCut_hh
#define tmEventSetup_L1TUtmCut_hh

/*====================================================================*
 * declarations
 *====================================================================*/
/*-----------------------------------------------------------------*
 * headers
 *-----------------------------------------------------------------*/
#include <string>
#include "CondFormats/Serialization/interface/Serializable.h"

#include "CondFormats/L1TObjects/interface/L1TUtmCutValue.h"


/*-----------------------------------------------------------------*
 * constants
 *-----------------------------------------------------------------*/
/* nope */


/**
 *  This class implements data structure for Cut
 */
class L1TUtmCut
{
  public:
    // ctor
    L1TUtmCut()
      : name_(), object_type_(), cut_type_(),
        minimum_(), maximum_(), data_() { };

    // dtor
    virtual ~L1TUtmCut() { };


    /** set cut name */
    void setName(const std::string& x) { name_ = x; };

    /** set object type */
    void setObjectType(const int x) { object_type_ = x; };

    /** set cut type */
    void setCutType(const int x) { cut_type_ = x; };

    /** set minimum value of cut range (double) */
    void setMinimum(const double x) { minimum_.value = x; };

    /** set minimum value of cut range (HW index) */
    void setMinimum(const unsigned int x) { minimum_.index = x; };

    /** set minimum value of cut range (L1TUtmCutValue struct) */
    void setMinimum(const L1TUtmCutValue& x) { minimum_ = x; };

    /** set maximum value of cut range (double) */
    void setMaximum(const double x) { maximum_.value = x; };

    /** set maximum value of cut range (HW index) */
    void setMaximum(const unsigned int x) { maximum_.index = x; };

    /** set maximum value of cut range (L1TUtmCutValue struct) */
    void setMaximum(const L1TUtmCutValue& x) { maximum_ = x; };

    /** get cut name */
    const std::string& getName() const { return name_; };

    /** get target object type : combination of esObjectType and esFunctionType enums */
    const int getObjectType() const { return object_type_; };

    /** get cut type */
    const int getCutType() const { return cut_type_; };

    /** get L1TUtmCutValue struct for minimum value of cut range */
    const L1TUtmCutValue& getMinimum() const { return minimum_; };

    /** get L1TUtmCutValue struct for maximum value of cut range */
    const L1TUtmCutValue& getMaximum() const { return maximum_; };

    /** get L1TUtmCutValue struct for minimum value of cut range */
    const double getMinimumValue() const { return minimum_.value; };

    /** get L1TUtmCutValue struct for maximum value of cut range */
    const double getMaximumValue() const { return maximum_.value; };

    /** get L1TUtmCutValue struct for minimum value of cut range */
    const unsigned int getMinimumIndex() const { return minimum_.index; };

    /** get L1TUtmCutValue struct for maximum value of cut range */
    const unsigned int getMaximumIndex() const { return maximum_.index; };

    /** get data */
    const std::string& getData() const { return data_; };

    /** get key */
    const std::string& getKey() const { return key_; };


  protected:
    std::string name_;          /**< name of cut */
    int object_type_;           /**< target object type */
    int cut_type_;              /**< type of cut */
    L1TUtmCutValue minimum_;        /**< minimum value of cut range */
    L1TUtmCutValue maximum_;        /**< maximum value of cut range */
    std::string data_;          /**< data for charge/quality/isolation/charge correlation */
    std::string key_;           /**< key for accessing a scale */
    unsigned int version;
  COND_SERIALIZABLE;
};

#endif // tmEventSetup_L1TUtmCut_hh
/* eof */
