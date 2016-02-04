function makeSelectRequest(requestURL, selectMenuName)                               
{                                                         // Unused?                           
  var selectMenu = document.getElementById(selectMenuName);                          
  var request = requestURL + "&Argument=" + selectMenu.value;                        
  WebLib.makeRequest(request, WebLib.dummy);                                                       
}
