#ifndef GlobalTriggerAnalyzer_L1GtPatternLine_h
#define GlobalTriggerAnalyzer_L1GtPatternLine_h

/**
 * \class L1GtPatternLine
 * 
 * 
 * Description: A representation of a of pattern file line for
 *              L1 GT hardware testing.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Thomas Themel - HEPHY Vienna
 * 
 *
 */

#include <string>
#include <map>
#include <cstdint>

/** A class representing the contents of one line in a pattern file. 
    The contents are represented as (name, value) pairs. Column ids
    are also enumerated so that multiple columns with identical names
    end up with consecutive names ("muon" becomes "muon1", "muon2" etc).
*/
class L1GtPatternLine {
public:
  /** Add a new column with the specified name prefix and value.
      @param prefix The name prefix of the new column. The final column name
                    consists of prefix + the lowest free number (starting at 1)
      @param value  The actual data content of the column.
  */
  void push(const std::string& prefix, uint32_t value);

  /** Manipulate an existing value. 
      @param name  the name (prefix + no) of the column to set. Do not use _h or _l values here!
      @param value the new value for the column.
  */
  void set(const std::string& name, uint32_t value);

  /** Debug dump of contents */
  void print(std::ostream& out) const;

  /** Returns true iff a column of the given name exists.
      @param colname Column name to look for. 
                     Beware: This has to include the number appended by push!
  */
  bool has(const std::string& colname) const;

  /** Returns the next free column name for a given prefix. */
  std::string nextName(const std::string& prefix);

  /** Forms a column name from a prefix and a number. */
  std::string name(const std::string& prefix, unsigned int i) const;

  /** Accessor. @see has*/
  uint32_t get(const std::string& name) const;

private:
  typedef std::map<std::string, uint32_t> ColumnMap;
  ColumnMap m_columns;
};

#endif /*GlobalTriggerAnalyzer_L1GtPatternLine_h*/
