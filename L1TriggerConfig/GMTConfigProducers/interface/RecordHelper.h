#ifndef CondTools_L1Trigger_RecordHelper_h
#define CondTools_L1Trigger_RecordHelper_h
// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     RecordHelper
//
/**\class RecordHelper RecordHerlper.h CondTools/L1Trigger/interface/RecordHelper.h

 Description: A microframework to deal with the need to fill rather boring getter/setter classes
              from Coral classes.


*/

#include <boost/type_traits.hpp>

#include "RelationalAccess/ICursor.h"
#include "CoralBase/AttributeList.h"
#include "CoralBase/AttributeSpecification.h"
#include "CoralBase/Attribute.h"

/** A C++ version of C's strupper */
std::string upcaseString(std::string aString);

/** A base class for all field handlers that create TOutput C++ objects. */
template <class TOutput>
class FieldHandlerBase {
public:
  typedef coral::AttributeList AttributeList;
  /** Construct a new field handler with the C++ field name as its argument */
  FieldHandlerBase(const std::string& name) : name_(name) {}

  /** Return the name of the field handled by this object. */
  const std::string& getName() { return name_; }

  /** Return the name of the associated database field. For the GMT database,
   *  this simply corresponds to the uppercased field name. */
  virtual const std::string getColumnName() { return upcaseString(name_); }

  /** The actual extraction function. src contains a CORAL attribute list representing a 
   *  query result row, dest is the output object to be filled. */
  virtual void extractValue(const AttributeList& src, TOutput& dest) = 0;

  /** Virtual destructor for children. */
  virtual ~FieldHandlerBase() {}

private:
  /* The (case sensitive!) name of the field. */
  std::string name_;
};

/** A template field handler that simply 
 *    - casts the data field to TCField
 *    - converts that value to TCField
 *    - passes it to a setter method of a TOutput object
 */
template <class TOutput, class TCField, class TDBField>
class FieldHandler : public FieldHandlerBase<TOutput> {
public:
  typedef coral::AttributeList AttributeList;
  typedef void (TOutput::*TSetMethod)(const TCField);

  FieldHandler(const std::string& fieldName, TSetMethod setter)
      : FieldHandlerBase<TOutput>(fieldName), setter_(setter) {}

  /** Actual data extraction. */
  void extractValue(const AttributeList& src, TOutput& dest) override {
#ifdef RECORDHELPER_DEBUG
    std::cout << "Parsing field " << this->getName() << " with type " << typeid(TCField).name();
#endif
    typedef typename boost::remove_cv<typename boost::remove_reference<TDBField>::type>::type TDBFieldT;
    const TDBFieldT& value = src[this->getColumnName()].template data<TDBFieldT>();
    ((dest).*setter_)(TCField(value));

#ifdef RECORDHELPER_DEBUG
    std::cout << "=" << TCField(value) << std::endl;
#endif
  }

protected:
  /** Points to the setter method used to stuff the field's value into the
     destination object. */
  TSetMethod setter_;
};

/** A special handler for bool fields in the GT/GMT DBs. These can't be imported
 *  in the generic way because bool values are returned as char '0' for false and 
 * '1' for true from the database. Basically, all values that are != FalseCharacter
 *  are treated as true (in adherence to the venerable C tradition).
 */
template <class TOutput, char FalseCharacter>
class ASCIIBoolFieldHandler : public FieldHandler<TOutput, bool, char> {
public:
  typedef coral::AttributeList AttributeList;
  ASCIIBoolFieldHandler(const std::string& fieldName, typename FieldHandler<TOutput, bool, char>::TSetMethod setter)
      : FieldHandler<TOutput, bool, char>(fieldName, setter) {}

  /** Extract value as char, then see compare it to '0' to get its truth value. */
  void extractValue(const AttributeList& src, TOutput& dest) override {
    char value = src[this->getColumnName()].template data<char>();
#ifdef RECORDHELPER_DEBUG
    std::cout << " .. and " << this->getColumnName() << " is (in integers) " << (int)value << std::endl;
#endif
    ((dest).*(this->setter_))(value != FalseCharacter);
  }
};

/** Tag class: Allow overriding of field type -> field handler type mapping
               according to the group that a type is added to. This means that
	       we can override the standard type mappings for some functions. */

/** The standard group: All types for which no explicit group is defined end up 
    here. */
class TStandardGroup;

/** Generic mapping - all types get TStandardGroup. */
template <typename TOutput>
struct Group {
  typedef TStandardGroup Type;
};

/** A macro that assigns a record type to a group. */
#define RH_ASSIGN_GROUP(TOutput, TGroup) \
  template <>                            \
  struct Group<TOutput> {                \
    typedef TGroup Type;                 \
  };

/** What is all this group stuff good for? It provides a layer of indirection
    between output types and field handlers that is not specific to each type,
    so that it's possible to handle a bunch of types with a uniform database column ->
    field structure mapping with just one group def + 1 RH_ASSIGN_GROUP per type.
*/

/* Generically: We don't mess with types and expect the field types to
   exactly match the database types (sadly, CORAL is rather anal about
   such things). */
template <typename TOutput, typename TGroup, typename TCType>
struct GroupFieldHandler {
  typedef FieldHandler<TOutput, TCType, TCType> Type;
};

template <class TOutput>
class RecordHelper {
public:
  typedef coral::AttributeList AttributeList;
  /** A list of field handlers that determine how to handle a record. */
  typedef std::vector<FieldHandlerBase<TOutput>*> FieldVector;

  template <typename TField>
  void addField(const std::string& fieldName, void (TOutput::*setter)(const TField)) {
#ifdef RECORDHELPER_DEBUG
    std::cout << "Adding field " << fieldName << ", type = " << typeid(TField).name() << std::endl;
#endif
    this->fields_.push_back(
        new typename GroupFieldHandler<TOutput, typename Group<TOutput>::Type, TField>::Type(fieldName, setter));
  }

  /** Iterates over all known fields and extracts the fields of the record
      in source to the object in dest. */
  virtual void extractRecord(const AttributeList& source, TOutput& dest) {
    for (typename FieldVector::const_iterator it = fields_.begin(); it != fields_.end(); ++it) {
      (*it)->extractValue(source, dest);
    }
  }

  /** Returns a list of database column names for the added fields. */
  virtual std::vector<std::string> getColumnList() {
    std::vector<std::string> colList;
    for (typename FieldVector::const_iterator it = fields_.begin(); it != fields_.end(); ++it) {
      colList.push_back((*it)->getColumnName());
    }

    return colList;
  }

  /** Destructor: Wipe out all the field handlers. */
  virtual ~RecordHelper() {
    for (typename FieldVector::iterator it = fields_.begin(); it < fields_.end(); ++it) {
      delete *it;
    }
  }

protected:
  /** List of known fields. TODO: Make private, add base type addField. */
  FieldVector fields_;
};

/** A helper macro to reduce the amount of typing, since
   the field name completely determines the setter. */
#define ADD_FIELD(HELPER, OUTPUT_NAME, FIELD_NAME) HELPER.addField(#FIELD_NAME, &OUTPUT_NAME::set##FIELD_NAME);

#endif
