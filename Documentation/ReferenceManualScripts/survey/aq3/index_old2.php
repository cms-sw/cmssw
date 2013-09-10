<?php

$pageTitle = "Index";

include_once("header.php");


function printPNInfo(){

echo "
<script type=\"text/javascript\">
    $(window).load(function(){
    $('#PNInfo').clickover({ placement: 'right'});
   });
</script>
";

echo "     
  <a href=\"#\" id='PNInfo' rel=\"clickover\" data-content=\"
	Starting with <b>CMS-</b> as in <a target='_blank' href='http://cdsweb.cern.ch/collection/CMS%20Papers?ln=en' class='btn btn-mini btn-info'>here</a>

   \" data-original-title=\"The CMS publication number <button type='button'class='close' data-dismiss='clickover'>&times;</button>\">
	<i class='icon-question-sign'></i></a>";
}


function printPasswordInfo(){

echo "
<script type=\"text/javascript\">
    $(window).load(function(){
    $('#passwordInfo').clickover({ placement: 'right'});
   });
</script>
";

echo "     
  <a href=\"#\" id='passwordInfo' rel=\"clickover\" data-content=\"
	<ul>
		<li>To save <b>your</b> answers.</li>
		<li>Make sure nobody else overwrites your answers.</li>
	</ul>

   \" data-original-title=\"Why do I need password? <button type='button'class='close' data-dismiss='clickover'>&times;</button>\">
	<i class='icon-question-sign'></i></a>";
}

function printEmailInfo(){

echo "
<script type=\"text/javascript\">
    $(window).load(function(){
    $('#emailInfo').clickover({ placement: 'right'});
   });
</script>
";

echo "     
  <a href=\"#\" id='emailInfo' rel=\"clickover\" data-content=\"
	<ul>
		<li>To identify survey's owner.</li>
		<li>Password will be sent to this email in case you forgot it.</li>
	</ul>

   \" data-original-title=\"Why do I need to provide email? <button type='button'class='close' data-dismiss='clickover'>&times;</button>\">
	<i class='icon-question-sign'></i></a>";
}

  echo "<center><h1>Analysis questionnaire</h1></center><hr/><br/>";

?>
<script>

$(document).ready(function(){
				
	$('#form').submit(function(){

	success = true;
	if (document.getElementById('password').value == '') {		
		document.getElementById('password').style.border='2px solid red';
		success = false; 
	}
	else{
		document.getElementById('password').style.border='2px solid green';
	} 

	if (document.getElementById('email').value == '' || !isEmail(document.getElementById('email').value)) {		
		document.getElementById('email').style.border='2px solid red';
		success = false; 
	}
	else{
		document.getElementById('email').style.border='2px solid green';
	}

	if (document.getElementById('publication_number').value == '') {		
		document.getElementById('publication_number').style.border='2px solid red';
		success = false; 
	}
	else{
		document.getElementById('publication_number').style.border='2px solid green';
	} 
		return success; 
	
    });
});

function isEmail(email) {
    var p = /^[a-zA-Z0-9._-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}$/;
    return p.test(email);
};

</script>

<style>
.popover{
	width:250px;
}
</style>

<div class="rounded" style='margin:0px auto; width:970px; '>
<div class="rounded newSurvey">
	<h3>Start new survey</h3><hr style="border-color:#5A7FAD"/>

	<form id="form" action="questionnaire.php" method="POST">
		Publication number <?php  printPNInfo(); ?><br/>
		<input type="text" name="publication_number" placeholder="CMS-XYZ-YY-nnn" id="publication_number"><br/>
		Email <?php  printEmailInfo(); ?><br/>
		<input type="text" name="email" id="email"><br/>
		Password <?php  printPasswordInfo(); ?> <br/>
		<input type="text" name="password"  id="password"><br/><br/>
		<input type='hidden' name="action" value="startNew">
		<input type="submit" value="Start new survey" class="btn btn-success">
	</form>
</div>

<div class="rounded" style='margin:10px; width:623px; border:1px solid black; text-align:center; background:#7A96B9;float:left'>
	<h3>Update survey</h3>

<?php

try
  {
    //open the database
    $db = new PDO('sqlite:aq.db');
    
    $questionnaires = $db->query('SELECT * FROM Questionnaire')->fetchAll();

	echo "<center><div style='width:623px'>
	";
	echo "<div style='border-bottom:1px solid black; border-top:1px solid black; padding-top:10px; padding-bottom:10px; background:#7A96B9'>
		<div class='qcol border_right'>Publication number</div> 
		<div class='qcol border_right'>Contact</div> 
		<div class='qcol100'>Action</div>
		<div style='clear:both'></div>
	      </div>
	<div style='overflow:scroll; overflow-x: hidden; overflow-y: auto; height:216px'>
	";

    foreach($questionnaires as $questionnaire)
    {
    	
	echo "<div class='qrow'>
		<div class='qcol border_right'>".$questionnaire['publication_number']."</div> 
		<div class='qcol border_right'>".str_replace("@", "<i class='icon-globe'></i>",$questionnaire['email'])."</div>
		<div class='qcol100'>			
			<a target='_blank' href='questionnaire.php?id=".$questionnaire['id']."' class='btn btn-mini btn-success'><i class='icon-wrench icon-white'></i> Update</a>
		</div>
		<div style='clear:both'></div>
	      </div>";
    }
	
	echo "
	</div>
	</div><br/>


</center>";	
    // close the database connection
    $db = NULL;
  }
  catch(PDOException $e)
  {
    print 'Exception : '.$e->getMessage();
  }
?>

</div>
<div style="clear:both"></div>

</div>


<center>
 <hr>

<script type="text/javascript" charset="utf-8">
// Javascript
$(function() {

  var currentDate = new Date();
  $('div#clock').countdown(new Date(2013, 0, 1), function(event) {
    $this = $(this);
    switch(event.type) {
      case "seconds":
      case "minutes":
      case "hours":
      case "days":
      case "weeks":
      case "daysLeft":
        $this.find('span#'+event.type).html(event.value);
        break;
      case "finished":
        $this.fadeTo('slow', .5);
        break;
    }
  });
});
</script>

		<h2>Time left to complete your survey</h2>
	<div id="clock">
	  <p class="rounded">
	    <span id="weeks"></span>
	    Weeks
	  </p>
	  <div class="space">:</div>
	  <p class="rounded">
	    <span id="daysLeft"></span>
	    Days
	  </p>
	  <div class="space">:</div>
	  <p class="rounded">
	    <span id="hours"></span>
	    Hours
	  </p>
	  <div class="space">:</div>
	  <p class="rounded">
	    <span id="minutes"></span>
	    Minutes
	  </p>
	  <div class="space">:</div>
	  <p class="rounded">
	    <span id="seconds"></span>
	    Seconds
	  </p>
	</div>

</center>


<?php
include_once("footer.php");
?>

