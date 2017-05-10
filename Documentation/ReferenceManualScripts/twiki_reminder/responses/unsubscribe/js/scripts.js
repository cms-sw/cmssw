$(document).ready(function() {
  $('#other').attr('checked', false);
	$('#other').click(function(){
  	if ($(this).is(':checked')){
  	  $('#othr').show();
  		$('#othr').animate({height:200, width:388});
  	}
  	else {
      $('#othr').animate({height:0, width:388});
      $('#othr').hide('slow');
    }
	});
});