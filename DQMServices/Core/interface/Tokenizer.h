#ifndef Tokenizer_h
#define Tokenizer_h
#include <string>
#include <vector>

/** Tokenize "input" in a vector<string> at each occurence of "sep"
 */
namespace dqm{
class Tokenizer : public std::vector<std::string> {
public:
  typedef std::vector<std::string> super;
  Tokenizer(const std::string & sep, const std::string & input, 
	    bool alsoempty=true);
  
  void join(std::string & out, const std::string & sep, bool alsoempty=true)
    const;

private:

  };
}
#endif // Tokenizer_h
