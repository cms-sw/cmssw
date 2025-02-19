function submitConfigure(url, myform)                                                
{                                                                                    
  navigator_form = false;                                                            
  url = url + "/Request";                                                            
  url = url + "?" + "RequestID=Configure";                                           
  url = url + "&" + "Hostname=" + myform.Hostname.value;                             
  url = url + "&" + "Port=" + myform.Port.value;                                     
  url = url + "&" + "Clientname=" + myform.Name.value;                               
                                                                                     
  var funct = alertContents;                                                         
  makeRequest(url, funct);                                                           
}                                                                                    
                                                                                     
//*************************************************************/                     
                                                                                     
function alertContents()                                                             
{                                                                                    
  if (http_request.readyState == 4)                                                  
  {                                                                                  
    if (http_request.status == 200)                                                  
    {                                                                                
      alert("Configuration Submitted");                                              
    }                                                                                
    else                                                                             
    {                                                                                
      alert('There was a problem with the request.');                                
    }                                                                                
  }                                                                                  
}