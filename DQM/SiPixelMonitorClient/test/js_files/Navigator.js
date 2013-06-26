var Navigator = {} ;

Navigator.navigator_current = "top";

//___________________________________________________________________________________
/*                                                                                   
  This function returns the URL that should be loaded as                             
  a result of clicks on the drop down menus of the navigator form.                   
*/                                                                                   
Navigator.getNavigatorRequestURL = function()                                                    
{                                                           // Unused?                         
  var form = document.getElementById("NavigatorForm");                               
  var open = form.Open;                                                              
  var subscribe   = form.Subscribe;                                                  
  var unsubscribe = form.Unsubscribe;                                                
                                                                                     
  url = WebLib.getApplicationURL();                                                         
                                                                                     
  if (open.value != "")                                                              
  {                                                                                  
    //url = url + "/Request?"                                                           
    url = url + "RequestID=Open";                                              
    url = url + "&" + "Current=" + Navigator.navigator_current;                                
    url = url + "&" + "Open=" + open.value;                                          
  }                                                                                  
  else if (subscribe.value != "")                                                    
  {                                                                                  
    //url = url + "/Request?";                                                          
    url = url + "RequestID=Subscribe";                                         
    url = url + "&" + "Current=" + Navigator.navigator_current;                                
    url = url + "&" + "SubscribeTo=" + subscribe.value;                              
  }                                                                                  
  else if (unsubscribe.value != "")                                                  
  {                                                                                  
    //url = url + "/Request?";                                                          
    url = url + "RequestID=Unsubscribe";                                       
    url = url + "&" + "Current=" + Navigator.navigator_current;                                
    url = url + "&" + "UnsubscribeFrom=" + unsubscribe.value;                        
  }                                                                                  
  return url;                                                                        
}                   

//___________________________________________________________________________________
/*                                                                                   
  This function updates the navigator drop down menus according                      
  to the xml of the server response.                                                 
*/                                                                                   
Navigator.updateNavigator = function()                                                           
{                                                           // Unused?                          
  if (WebLib.http_request.readyState == 4)                                                  
  {                                                                                  
    if (WebLib.http_request.status == 200)                                                  
    {                                                                                
      var xmldoc;                                                                    
      var subdirs_l;                                                                 
      var subscribe_l;                                                               
      var unsubscribe_l;                                                             
                                                                                     
      // Load the xml elements on javascript lists:                                  
      if (WebLib.http_request != false)                                                     
      {                                                                              
        xmldoc = WebLib.http_request.responseXML;                                           
        Navigator.navigator_current = xmldoc.getElementsByTagName('current').item(0).firstChild.data;
        subdirs_l     = xmldoc.getElementsByTagName('open');                             
        subscribe_l   = xmldoc.getElementsByTagName('subscribe');                      
        unsubscribe_l = xmldoc.getElementsByTagName('unsubscribe');                  
      }                                                                              
                                                                                     
      var form = document.getElementById("NavigatorForm");                           
      var open = form.Open;                                                          
      var subscribe   = form.Subscribe;                                              
      var unsubscribe = form.Unsubscribe;                                            
                                                                                     
      // Update the Open menu:                                                       
      open.options.length = 0;                                                       
                                                                                     
      open.options[0] = new Option("", "", true, true);                              
      open.options[1] = new Option("top", "top", false, false);                      
      for(var i = 0; i < subdirs_l.length; i++)                                      
      {                                                                              
        var to_open = subdirs_l.item(i).firstChild.data;                             
        open.options[i + 2] = new Option(to_open, to_open, false, false);            
      }                                                                              
      open.selectedIndex = 0; 

      // Update the Subscribe menu:                                                  
      subscribe.options.length = 0;                                                  
      subscribe.options[0] = new Option("", "", true, true);                         
      for(var i = 0; i < subscribe_l.length; i++)                                    
      {                                                                              
        var to_subscribe = subscribe_l.item(i).firstChild.data;                      
        subscribe.options[i + 1] = new Option(to_subscribe, to_subscribe, false, false);
      }                                                                              
      subscribe.selectedIndex = 0;                                                   
                                                                                     
      // Update the Unsubscribe menu:                                                
      unsubscribe.options.length = 0;                                                
      unsubscribe.options[0] = new Option("", "", true, true);                       
      for(var i = 0; i < unsubscribe_l.length; i++)                                  
      {                                                                              
        var to_unsubscribe = unsubscribe_l.item(i).firstChild.data;                  
        unsubscribe.options[i + 1] = new Option(to_unsubscribe, to_unsubscribe, false, false);
      }                                                                              
      unsubscribe.selectedIndex = 0;                                                 
    }                                                                                
  }                                                                                  
}                                                                                    
                                                                                     
//___________________________________________________________________________________
Navigator.makeNavigatorRequest = function()                                                      
{                                                            // Unused?                        
  url = Navigator.getNavigatorRequestURL();                                                    
                                                                                     
  // pass a reference to the updateNavigator function:                               
  WebLib.makeRequest(url, Navigator.updateNavigator);                                                 
}                    

