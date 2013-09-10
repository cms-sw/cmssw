<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">

<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">
<head>
<meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />
<title></title>
<link href="css/screen.css" rel="stylesheet" type="text/css"  />
<link href="css/style.css" rel="stylesheet" type="text/css"  />

<script type="text/javascript" src="js/jquery.js"></script>
<script type="text/javascript" src="js/scripts.js"></script>

</head>
   <body>
    <div id="patternTopBar"></div>
    <a id="logo" href="https://twiki.cern.ch"><img src="images/outline-blue2.png" /></a>
    <div class="container">
      <form action="confirm_form.php" method="post" class="uns_form">
      <h2>Please fill the form below:</h2>
        <label>E-mail:</label>
        <input type="text" name="email" value="<?php echo $_GET['sender']; ?>">
        <label>Reason</label>
        <select name="reason">
	  <option value="">---</option>    
          <option value="Not working in CERN anymore">Not working at CERN anymore</option>
          <option value="Changed department">Changed department</option>          
        </select>
        <input type="checkbox" id="other" />
        <label class="inline" for="other">Other ...</label>
        <div class="clear"></div>
        <textarea id="othr" style="height:0;display:none;" cols="1" rows="1" name="other"></textarea>     
        <div class="clear"></div>  
        <input type="submit" value="Send">
      </form>
    </div>
  </body>
</html>
