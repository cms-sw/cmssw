#ifndef DD_DDStreamer_h
#define DD_DDStreamer_h

#include <iostream>
class DDCompactView;

//! Streaming the DDD transient store from/into a std::istream/std::ostream */
/**
  DDStreamer can be used to write the DDD transient object into a std::ostram and
  to retrieve them again via an std::istream.

  <br>
  The streamer can also be used together with DDLParser. Once possible usage scenario
  would be to load most of the geometrical DDD information via the streamer and
  parse some addional DDD XML documents containing SpecPar-information using DDLParser.
  
  <br>
  If DDStreamer is used together with DDLParser, the user has to ensure that reading in
  via DDStreamer::read() is done BEFORE invoking DDLParser to guarantee internal consistensies
  of the DDD objects.
  
  <br>
  
  <br>
  \code
  // writing:
  #include<fstream>
  std::ofstream file("pers.txt");
  DDStreamer streamer(file);
  streamer.write();
  
  
  // reading:
  #include<fstream>
  std::ifstream file("pers.txt");
  DDStreamer streamer(filer);
  streamer.read();
  \endcode
  
*/
class DDStreamer
{
public:
  //! constructs a streamer object with yet undefined std::istream and std::ostream
  DDStreamer();
  
  //! creates a streamer object for reading
  DDStreamer(std::istream & readFrom);
  
  //! creates a streamer object for writing
  DDStreamer(std::ostream & writeTo);
  
  //! does nothing; usefull only if another streamer derives from DDStreamer
  virtual ~DDStreamer(); 
  
  //! stream all DDD transient objects to the std::ostream referred to by member o_
  void write();
  
  //! populate DDD transient objects from the std::istream refetrred to by member i_
  void read();
  
  //! stream all DDD transient objects to the given std::ostream os
  void write(std::ostream & os);
  
  //! populate DDD transient objects from the given std::istream is
  void read(std::istream & is);
  
  //! set the istream for DDStreamer::read()
  void setInput(std::istream & i) { i_ = &i; }
  
  //! set the std::ostream for DDStreamer::write()
  void setOutput(std::ostream & o) { o_ = &o; }
  
protected:  
  //! write all instances of DDName
  void names_write();
  
  //! read all instances of DDName
  void names_read();
  
  //! write all instances of DDMaterial
  void materials_write();
  
  //! read all instances of DDMaterial
  void materials_read();
  
  //! write all instances of DDSolid
  void solids_write();
  
  //! read all instances of DDSolid
  void solids_read();

  //! write all instances of DDLogicalPart
  void parts_write();
  
  //! read all instances of  DDLogicalPart
  void parts_read();
  
  //! write all instances of DDRotation
  void rots_write();
  
  //! read all instances of DDRotation
  void rots_read();
  
  //! write the graph structure of DDCompactView::graph()
  void pos_write();
  
  //! read the graph structure for DDCompactView::graph()
  void pos_read();
  
  //! write all instances of DDSpecifics
  void specs_write();
  
  //! read all instances of 
  void specs_read(); 
  
  //! write the dictionary of ClhepEvaluator
  void vars_write();
  
  //! read the dictionlary of ClhepEvaluator
  void vars_read();  

private:  
  std::ostream * o_; /**< std::ostream target for writing DDD objects */
  std::istream * i_; /**< istream target for reading DDD objects */
};
#endif
