function displayMessages()                                                       
{                                                                                    
  if (http_request.readyState == 4)                                                  
  {                                                                                  
    if (http_request.status == 200)                                                  
    {                                                                                
      var xmldoc;                                                                    
                                                                                     
      // Load the xml elements on javascript lists:                                  
      if (http_request != false)                                                     
      {                                                                              
        xmldoc = http_request.responseXML;                                           
                                                                                     
        // set the contentViewer_current first:                                      
        types_l  = xmldoc.getElementsByTagName('Type');
        titles_l = xmldoc.getElementsByTagName('Title');                             
        texts_l  = xmldoc.getElementsByTagName('Text');                                
      }                                               

      for (var i = 0; i < types_l.length; i++)
      {
        alert("MESSAGE: " + titles_l.item(i).firstChild.data + "\n" + "TYPE: " + types_l.item(i).firstChild.data + "\n" + texts_l.item(i).firstChild.data); 
      }
    }                                                                                
  }                                                                                  
}                                                                                    
