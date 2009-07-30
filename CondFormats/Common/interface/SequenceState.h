#ifndef Cond_SequenceState_h
#define Cond_SequenceState_h

namespace cond {
  
  class IOVSequence;
  
  /** capture the state of a IOV sequence at a given time
   */
  class SequenceState {
  public:
    ~SequenceState();
    SequenceState();
    explicit SequenceState(IOVSequence const & seq);

    unsigned int size() const { return m_size; }
    
    int revision() const { return  m_revision;}
    
    cond::Time_t timestamp() const { return m_timestamp;}
    
    std::string const & comment() const  { return m_comment;}
  

  private:
    
    unsigned int m_size;
    int m_revision;
    cond::Time_t m_timestamp;
    std::string m_comment;
    
  };



#endif
