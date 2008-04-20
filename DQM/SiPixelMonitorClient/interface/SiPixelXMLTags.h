// +--[ STRING view: complete summary ]--------------------------------------+
// |									     |
// |  --[ WARNINGS ]------------------------------------------------------   |
// |  <path to module>  						     |
// |	<QTest Name>							     |
// |	<QTest Warning Message> 					     |
// |  ...								     |
// |  <path to module>  						     |
// |	<QTest Name>							     |
// |	<QTest Warning Message> 					     |
// |  --------------------------------------------------------------------   |
// |									     |
// |  --[ ERRORS ]--------------------------------------------------------   |
// |  <path to module>  						     |
// |	<QTest Name>							     |
// |	<QTest Error Message>						     |
// |  ...								     |
// |  <path to module>  						     |
// |	<QTest Name>							     |
// |	<QTest Error Message>						     |
// |  --------------------------------------------------------------------   |
// |									     |
// +-------------------------------------------------------------------------+
//
// +--[ STRING view: lite summary ]------------------------------------------+
// |									     |
// |  --[ WARNINGS ]------------------------------------------------------   |
// |	Digis	: <# of modules with Digis Warnings> of <Tot Modules #>      |
// |	Clusters: <# of modules with Clusters Warnings> of <Tot Modules #>   |
// |  --------------------------------------------------------------------   |
// |									     |
// |  --[ ERRORS ]--------------------------------------------------------   |
// |	Digis	: <# of modules with Digis Errors> of <Tot Modules #>	     |
// |	Clusters: <# of modules with Clusters Errors> of <Tot Modules #>     |
// |  --------------------------------------------------------------------   |
// |									     |
// +-------------------------------------------------------------------------+
//
// +--[ XML view: complete summary ]-----------------------------------------+
// |									     |
// |  <warnings>							     |
// |	<module>							     |
// |	  <path>...</path>						     |
// |	  <qtest name='...'>Message</qtest>				     |
// |	  ...								     |
// |	  <qtest name='...'>Message</qtest>				     |
// |	</module>							     |
// |	...								     |
// |	<module>							     |
// |	  <path>...</path>						     |
// |	  <qtest name='...'>Message</qtest>				     |
// |	  ...								     |
// |	  <qtest name='...'>Message</qtest>				     |
// |	</module>							     |
// |  </warnings>							     |
// |  <errors>  							     |
// |	<module>							     |
// |	  <path>...</path>						     |
// |	  <qtest name='...'>Message</qtest>				     |
// |	  ...								     |
// |	  <qtest name='...'>Message</qtest>				     |
// |	</module>							     |
// |	...								     |
// |	<module>							     |
// |	  <path>...</path>						     |
// |	  <qtest name='...'>Message</qtest>				     |
// |	  ...								     |
// |	  <qtest name='...'>Message</qtest>				     |
// |	</module>							     |
// |  </errors> 							     |
// |									     |
// +-------------------------------------------------------------------------+
//
// +--[ XML view: lite summary ]---------------------------------------------+
// |									     |
// |  <warnings>							     |
// |	<digis>Message</digis>  					     |
// |	<clusters>Message</clusters>					     |
// |	...								     |
// |  </warnings>							     |
// |  <errors>  							     |
// |	<digis>Message</digis>  					     |
// |	<clusters>Message</clusters>					     |
// |	...								     |
// |  </errors> 							     |
// |									     |
// +-------------------------------------------------------------------------+

// Author : Samvel Khalatian ( samvel at fnal dot gov)
// Created: 04/05/07

#ifndef DQM_SIPIXELMONITORCLIENT_SIPIXELXMLTAGS_H
#define DQM_SIPIXELMONITORCLIENT_SIPIXELXMLTAGS_H

#include <ostream>
#include <string>
#include <vector>

namespace dqm {
  /** 
  * @brief 
  *   Base for all XMLTags
  */
  class XMLTag {
    public:
      enum TAG_MODE { XML,
		      XML_LITE,
		      STRING,
		      STRING_LITE };

      XMLTag(): eTagMode_( XML) {}
      virtual ~XMLTag();

      /** 
      * @brief 
      *   Current mode will affect helper operator << that is specified for
      *   XML Tag.
      * 
      * @param roNEW_TAG_MODE  Obvious
      */
      inline void setMode( const TAG_MODE &roNEW_TAG_MODE) {
	eTagMode_ = roNEW_TAG_MODE;
      }

      inline TAG_MODE getMode() const { return eTagMode_; }

      /** 
      * @brief 
      *   Objects are created in free store: don't remove them amnually.
      *   Memory will be automatically cleaned up by XMLTag. 
      * 
      * @return 
      */
      template<class T>
	T *createChild();

      virtual std::ostream &printXML	   ( std::ostream &roOut) const;
      virtual std::ostream &printXMLLite   ( std::ostream &roOut) const;
      virtual std::ostream &printString    ( std::ostream &roOut) const;
      virtual std::ostream &printStringLite( std::ostream &roOut) const;

    private:
      typedef std::vector<const XMLTag *> VXMLTags;
      
      // Prevent copying
      XMLTag( const XMLTag &);
      XMLTag &operator ==( const XMLTag &);

      // [Note: all children are supposed to be created with *new* operator and
      //	will be automatically removed in destructor]
      VXMLTags oVTags_;
      TAG_MODE eTagMode_;
  };

  template<class T>
    T *XMLTag::createChild() {

    T *poT( new T);
    // Check if requested class is actually derived from XMLTag
    if( dynamic_cast<XMLTag *>( poT)) {
      oVTags_.push_back( poT);
    } else {
      delete poT;
      poT = 0;
    }

    return poT;
  }

  std::ostream &operator <<( std::ostream &roOut, const XMLTag &roXML_TAG);

  /** 
  * @brief 
  *   QTest XML tag
  */
  class XMLTagQTest: public XMLTag {
    public:
      XMLTagQTest(): cName_   ( ""),
		     cMessage_( "") {}

      void setName   ( const std::string &rcNAME)    { cName_ = rcNAME; }
      void setMessage( const std::string &rcMESSAGE) { cMessage_ = rcMESSAGE; }

      virtual std::ostream &printXML   ( std::ostream &roOut) const;
      virtual std::ostream &printString( std::ostream &roOut) const;

    private:
      std::string cName_;
      std::string cMessage_;
  };

  /** 
  * @brief 
  *   Path XML Tag
  */
  class XMLTagPath: public XMLTag {
    public:
      XMLTagPath(): cPath_( "") {}

      void setPath( const std::string &rcPATH) { cPath_ = rcPATH; }

      virtual std::ostream &printXML   ( std::ostream &roOut) const;
      virtual std::ostream &printString( std::ostream &roOut) const;

    private:
      std::string cPath_;
  };

  /** 
  * @brief 
  *   Module XML Tag
  */
  class XMLTagModule: public XMLTag {
    public:
      virtual std::ostream &printXML   ( std::ostream &roOut) const;
      virtual std::ostream &printString( std::ostream &roOut) const;
  };

  /** 
  * @brief 
  *   Simply useful and handly lock class
  */
  class Safe {
    public:
      Safe( const bool bLOCKED = false): bLocked_( bLOCKED) {}
      virtual ~Safe() {}

      virtual void lock  () { bLocked_ = true;  }
      virtual void unlock() { bLocked_ = false; }

      // isLocked operator
      operator bool() const { return bLocked_; }

    private:
      bool bLocked_;
  };

  /** 
  * @brief 
  *   Manage Module count where Errors/Warnings appeared
  */
  class Modules: public Safe {
    public:
      Modules(): nModules_( 0),
		 nTotModules_( 0) {}

      virtual Modules &operator++();

      inline void setTotModules( const int &rnTOT_MODULES) {
	nTotModules_ = rnTOT_MODULES; }

      inline int getModules   () const { return nModules_;    }
      inline int getTotModules() const { return nTotModules_; }
    
    private:
      int nModules_;
      int nTotModules_;
  };

  /** 
  * @brief 
  *   Summary Lite Digis XML Tag
  */
  class XMLTagDigis: public XMLTag,
		     public Modules {
    public:
      virtual std::ostream &printXMLLite   ( std::ostream &roOut) const;
      virtual std::ostream &printStringLite( std::ostream &roOut) const;
  };

  /** 
  * @brief 
  *   Summary Lite Clusters XML Tag
  */
  class XMLTagClusters: public XMLTag,
			public Modules {
    public:
      virtual std::ostream &printXMLLite   ( std::ostream &roOut) const;
      virtual std::ostream &printStringLite( std::ostream &roOut) const;
  };

  /** 
  * @brief 
  *   Warnings XML Tag
  */
  class XMLTagWarnings: public XMLTag {
    public:
      virtual std::ostream &printXML	   ( std::ostream &roOut) const;
      virtual std::ostream &printXMLLite   ( std::ostream &roOut) const;
      virtual std::ostream &printString    ( std::ostream &roOut) const;
      virtual std::ostream &printStringLite( std::ostream &roOut) const;
  };

  /** 
  * @brief 
  *   Errors XML Tag
  */
  class XMLTagErrors: public XMLTag {
    public:
      virtual std::ostream &printXML	   ( std::ostream &roOut) const;
      virtual std::ostream &printXMLLite   ( std::ostream &roOut) const;
      virtual std::ostream &printString    ( std::ostream &roOut) const;
      virtual std::ostream &printStringLite( std::ostream &roOut) const;
  };
} // End namespace dqm

#endif // DQM_SIPIXELMONITORCLIENT_SIPIXELXMLTAGS_H
