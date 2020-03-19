#ifndef GlobalTriggerAnalyzer_L1GtPatternWriter_h
#define GlobalTriggerAnalyzer_L1GtPatternWriter_h

/**
 * \class L1GtPatternWriter
 * 
 * 
 * Description: Formats L1GtPatternMaps into text files.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Thomas Themel - HEPHY Vienna
 * 
 *
 */

#include <iosfwd>
#include <string>
#include <vector>
#include <cstdint>

class L1GtPatternMap;
class L1GtPatternLine;

/** The L1GtPatternWriter object is responsible for the actual formatting 
    of the content of one or more L1GtPatternMaps into a text file. */
class L1GtPatternWriter {
public:
  /** Construct a new pattern writer. 
   *  @param destination  output stream to write to
   *  @param header       string to be written before pattern lines
   *  @param footer       string to be written after pattern lines
   *  @param columns      vector of column names for each pattern line
   *  @param length       vector of column lengths (in bits!) for each pattern line
   *  @param defaults     vector of default values (written if the pattern data does not contain entries
   *                      for the column). Indexed like columns.
   *  @param bx           vector of bunch crossing numbers to print (empty for 'all')
   *  @param debug        set to true to enable extensive debug logging
   */
  L1GtPatternWriter(std::ostream& destination,
                    const std::string& header,
                    const std::string& footer,
                    const std::vector<std::string>& columns,
                    const std::vector<uint32_t>& lengths,
                    const std::vector<uint32_t>& defaultValues,
                    const std::vector<int>& bx,
                    bool debug = false);

  /** Write the lines from a pattern map to the output stream. */
  void writePatterns(const L1GtPatternMap& patterns);

  /** Format a single line. */
  virtual void writePatternLine(const L1GtPatternLine& line);

  /** Close the output stream */
  void close();

  virtual ~L1GtPatternWriter();

protected:
  /** Returns an and-mask to truncate an uint32_t to a specified
      length. */
  static uint32_t mask(uint32_t length);

private:
  std::ostream& m_dest;
  std::string m_header;
  std::string m_footer;
  std::vector<std::string> m_columns;
  std::vector<uint32_t> m_lengths;
  std::vector<uint32_t> m_defaults;
  std::vector<int> m_bx;
  bool m_debug;

  uint32_t m_lineNo;
};

#endif /*GlobalTriggerAnalyzer_L1GtPatternWriter_h*/
