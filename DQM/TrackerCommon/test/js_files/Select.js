function makeSelectRequest(requestURL, selectMenuName)                               
{                                                                                    
  var selectMenu = document.getElementById(selectMenuName);                          
  var request = requestURL + "&Argument=" + selectMenu.value;                        
  makeRequest(request, dummy);                                                       
}
