<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
<meta http-equiv="Access-Control-Allow-Origin" content="*"> 
<title>
Publication/Author/Institute
</title>

<link href="css/bootstrap.css" rel="stylesheet" type="text/css" media="all" />
<link href="css/bootstrap-responsive.css" rel="stylesheet" type="text/css" media="all" />

<script src="http://code.jquery.com/jquery-1.8.2.js"></script>
<script type="text/javascript" src="js/bootstrap.js"></script>
<script type="text/javascript" src="js/bootstrapx-clickover.js"></script>
<script src="js/jquery.countdown.js" type="text/javascript" charset="utf-8"></script>
<script type="text/javascript" src="https://www.google.com/jsapi"></script>

<script>
$(function ()
{ $("#edit").popover();
});

function changeYear(year){

	var url = document.URL;
	var newUrl = "";
	
	var baseUrl = url.substring(0, url.indexOf("?")+1);
	var params = url.substring(url.indexOf("?")+1);
	var parts = params.split("&");
	var yearChanged = false;
	
	for(var i = 0; i < parts.length; i++){
		var part = parts[i];
		if (part.indexOf("year=") !== -1){
			yearChanged = true;
			if (year != ""){
				part = "year="+year;
			}
			else{
				part = "";
			}
		}

		if (newUrl != "" && part != "" ){
			newUrl += "&";
		}
		newUrl += part;		
	}
	if (!yearChanged && year != ""){
		newUrl+="&year="+year;
	}

window.location =  baseUrl+newUrl;
}

function update(){

	

}

</script>

</head>
<body>
