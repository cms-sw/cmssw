#ifndef FWCore_ParameterSet_DocFormatHelper_h
#define FWCore_ParameterSet_DocFormatHelper_h

// Used internally by ParameterSetDescription in its
// print function. This function is used to produce
// human readable documentation.

#include <string>
#include <iosfwd>

namespace edm {

  class ParameterDescriptionNode;

  class DocFormatHelper {

  public:

    enum DescriptionParent {
      TOP,
      OR,
      XOR,
      AND,
      OTHER      
    };

    DocFormatHelper() :
      brief_(false),
      lineWidth_(80),
      indentation_(4),
      startColumn2_(24U),
      section_(),
      pass_(0),
      column1_(0),
      column2_(0),
      column3_(0),
      counter_(0),
      parent_(OTHER)
    { }

    void init();

    bool brief() const { return brief_; }
    size_t lineWidth() const { return lineWidth_; }
    int indentation() const { return indentation_; }
    int startColumn2() const { return startColumn2_; }

    void setBrief(bool value) { brief_ = value; }
    void setLineWidth(size_t value) { lineWidth_ = value; }
    void setIndentation(int value) { indentation_ = value; }

    std::string const& section() const { return section_; }
    void setSection(std::string const& value) { section_ = value; }

    int pass() const { return pass_; }
    void setPass(int value) { pass_ = value; }

    size_t column1() const { return column1_; }
    size_t column2() const { return column2_; }
    size_t column3() const { return column3_; }

    void setAtLeast1(size_t width) { if (width > column1_) column1_ = width; }
    void setAtLeast2(size_t width) { if (width > column2_) column2_ = width; }
    void setAtLeast3(size_t width) { if (width > column3_) column3_ = width; }

    int counter() const { return counter_; }
    void setCounter(int value) { counter_ = value; }
    void incrementCounter() { ++counter_; }
    void decrementCounter() { --counter_; }

    DescriptionParent parent() const { return parent_; }
    void setParent(DescriptionParent value) { parent_ = value; }

    size_t commentWidth() const;

    static void wrapAndPrintText(std::ostream & os,
                                 std::string const& text,
                                 size_t indent,
                                 size_t suggestedWidth);

    void indent(std::ostream & os) const;
    void indent2(std::ostream & os) const;

    static int offsetModuleLabel() { return 2; }
    static int offsetTopLevelPSet() { return 2; }
    static int offsetSectionContent() { return 4; }

  private:

    bool brief_;
    size_t lineWidth_;
    int indentation_;
    size_t startColumn2_;

    std::string section_;

    int pass_;

    size_t column1_;
    size_t column2_;
    size_t column3_;

    int counter_;

    DescriptionParent parent_;
  };
}
#endif
