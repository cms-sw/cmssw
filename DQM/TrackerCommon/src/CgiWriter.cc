# include "DQM/TrackerCommon/interface/CgiWriter.h"

void CgiWriter::output_preamble()
{
  out->getHTTPResponseHeader().addHeader("Content-Type", "text/html");
  out->getHTTPResponseHeader().addHeader("Expires", "0");
  out->getHTTPResponseHeader().addHeader("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0");
  out->getHTTPResponseHeader().addHeader("Cache-Control", "post-check=0, pre-check=0");
  out->getHTTPResponseHeader().addHeader("Pragma", "no-cache");

  *out << cgicc::HTMLDoctype(cgicc::HTMLDoctype::eStrict) << std::endl;
  // This is here for the sake of producing correct html:
  //  *out << cgicc::html() << std::endl;
  *out << cgicc::html().set("lang", "en").set("dir","ltr") << std::endl;
}

void CgiWriter::output_head()
{
  std::string js_file_url  = contextURL + "/temporary/WebLib.js";
  std::string css_file_url = contextURL + "/temporary/style.css";

  *out << cgicc::head() << std::endl;

  *out << cgicc::script().set("src", js_file_url.c_str()) << cgicc::script() << std::endl;

  *out << cgicc::link().set("type", "text/css").set("href", css_file_url.c_str()).set("rel", "stylesheet") << std::endl;

  *out << cgicc::meta().set("http-equiv", "pragma").set("content", "no-cache") << std::endl;

  *out << cgicc::head() << std::endl;
}

void CgiWriter::output_finish()
{
  *out << "</html>" << std::endl;
}
