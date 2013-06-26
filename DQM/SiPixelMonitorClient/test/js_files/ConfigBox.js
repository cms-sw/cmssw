var ConfigBox = {} ;
                                                                                    
//___________________________________________________________________________________
ConfigBox.submitConfigure = function(url, myform)                                              
{                                                   // Unused?                                   
  navigator_form = false;                                                            
  //url = url + "/Request?";                                                            
  url = url + "RequestID=Configure";                                           
  url = url + "&" + "Hostname=" + myform.Hostname.value;                             
  url = url + "&" + "Port=" + myform.Port.value;                                     
  url = url + "&" + "Clientname=" + myform.Name.value;                               
                                                                                     
  WebLib.makeRequest(url, ConfigBox.alertContents);                                                           
}                                                                                    
                                                                                                                                                                          
//___________________________________________________________________________________
ConfigBox.alertContents = function()                                                             
{                                                  // Unused?                                           
  if (WebLib.http_request.readyState == 4)                                                  
  {                                                                                  
    if (WebLib.http_request.status == 200)                                                  
    {                                                                                
      alert("[ConfigBox.alertContents] Configuration Submitted");                                              
    }                                                                                
    else                                                                             
    {                                                                                
      alert('[ConfigBox.alertContents] There was a problem with the request.');                                
    }                                                                                
  }                                                                                  
}
