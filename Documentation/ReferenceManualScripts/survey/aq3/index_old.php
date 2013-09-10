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

   \" data-original-title=\"The CMS publication number\">
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

   \" data-original-title=\"Why do I need password?\">
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
		<li>To identify questionnaire's owner.</li>
		<li>Password will be sent to this email in case you forgot it.</li>
	</ul>

   \" data-original-title=\"Why do I need to provide email?\">
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

<div class="rounded" style='margin:0px auto; width:300px; border:1px solid black; text-align:center; background:#7A96B9;'>
	<h3>Take your analysis survey</h3><hr style="border-color:#5A7FAD"/>

	<form id="form" action="questionnaire.php" method="POST">
		Publication number <?php  printPNInfo(); ?><br/>
		<input type="text" name="publication_number" placeholder="CMS-XYZ-YY-nnn" id="publication_number"><br/>
		Email <?php  printEmailInfo(); ?><br/>
		<input type="text" name="email" id="email"><br/>
		Password <?php  printPasswordInfo(); ?> <br/>
		<input type="text" name="password"  id="password"><br/><br/>
		<input type='hidden' name="action" value="startNew">
		<input type="submit" value="Start" class="btn btn-success">
	</form>
</div>



<a href="editor.php" class="btn btn-warning">Editor</a> 
<hr/>
<center><h3>Analysis questionnaire replies</h3></center>
<?php

try
  {
    //open the database
    $db = new PDO('sqlite:aq.db');
    
    $questionnaires = $db->query('SELECT * FROM Questionnaire')->fetchAll();

	echo "<center><div class='rounded' style='border:1px solid black; width:753px'>
	";
	echo "<div class='rounded' style='border-bottom:1px solid black; padding-top:10px; padding-bottom:10px; background:#7A96B9'>
		<div class='qcol border_right'>Publication number</div> 
		<div class='qcol border_right'>Contact</div> 
		<div class='qcol'>Action</div>
		<div style='clear:both'></div>
	      </div>";

    foreach($questionnaires as $questionnaire)
    {
    	
	echo "<div class='qrow'>
		<div class='qcol border_right'>".$questionnaire['publication_number']."</div> 
		<div class='qcol border_right'>".str_replace("@", "<i class='icon-globe'></i>",$questionnaire['email'])."</div>
		<div class='qcol'>
			<a href='questionnaire.php?id=".$questionnaire['id']."' class='btn btn-mini btn-primary'><i class='icon-eye-open icon-white'></i> View</a>
			<a href='questionnaire.php?id=".$questionnaire['id']."' class='btn btn-mini btn-success'><i class='icon-wrench icon-white'></i> Edit</a>
		</div>
		<div style='clear:both'></div>
	      </div>";
    }
	
	echo "</div><br/>
<a href='statistics.php' class='btn btn-primary'>Statistics</a>

</center>";	
    // close the database connection
    $db = NULL;
  }
  catch(PDOException $e)
  {
    print 'Exception : '.$e->getMessage();
  }


include_once("footer.php");
?>