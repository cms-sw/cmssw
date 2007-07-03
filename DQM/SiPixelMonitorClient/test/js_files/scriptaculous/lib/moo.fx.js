//(c) 2006 Valerio Proietti (http://mad4milk.net). MIT-style license.
//moo.fx.js - depends on prototype.js OR prototype.lite.js
//version 2.0

var Fx = fx = {};

Fx.Base = function(){};
Fx.Base.prototype = {

	setOptions: function(options){
		this.options = Object.extend({
			onStart: function(){},
			onComplete: function(){},
			transition: Fx.Transitions.sineInOut,
			duration: 500,
			unit: 'px',
			wait: true,
			fps: 50
		}, options || {});
	},

	step: function(){
		var time = new Date().getTime();
		if (time < this.time + this.options.duration){
			this.cTime = time - this.time;
			this.setNow();
		} else {
			setTimeout(this.options.onComplete.bind(this, this.element), 10);
			this.clearTimer();
			this.now = this.to;
		}
		this.increase();
	},

	setNow: function(){
		this.now = this.compute(this.from, this.to);
	},

	compute: function(from, to){
		var change = to - from;
		return this.options.transition(this.cTime, from, change, this.options.duration);
	},

	clearTimer: function(){
		clearInterval(this.timer);
		this.timer = null;
		return this;
	},

	_start: function(from, to){
		if (!this.options.wait) this.clearTimer();
		if (this.timer) return;
		setTimeout(this.options.onStart.bind(this, this.element), 10);
		this.from = from;
		this.to = to;
		this.time = new Date().getTime();
		this.timer = setInterval(this.step.bind(this), Math.round(1000/this.options.fps));
		return this;
	},

	custom: function(from, to){
		return this._start(from, to);
	},

	set: function(to){
		this.now = to;
		this.increase();
		return this;
	},

	hide: function(){
		return this.set(0);
	},

	setStyle: function(e, p, v){
		if (p == 'opacity'){
			if (v == 0 && e.style.visibility != "hidden") e.style.visibility = "hidden";
			else if (e.style.visibility != "visible") e.style.visibility = "visible";
			if (window.ActiveXObject) e.style.filter = "alpha(opacity=" + v*100 + ")";
			e.style.opacity = v;
		} else e.style[p] = v+this.options.unit;
	}

};

Fx.Style = Class.create();
Fx.Style.prototype = Object.extend(new Fx.Base(), {

	initialize: function(el, property, options){
		this.element = $(el);
		this.setOptions(options);
		this.property = property.camelize();
	},

	increase: function(){
		this.setStyle(this.element, this.property, this.now);
	}

});

Fx.Styles = Class.create();
Fx.Styles.prototype = Object.extend(new Fx.Base(), {

	initialize: function(el, options){
		this.element = $(el);
		this.setOptions(options);
		this.now = {};
	},

	setNow: function(){
		for (p in this.from) this.now[p] = this.compute(this.from[p], this.to[p]);
	},

	custom: function(obj){
		if (this.timer && this.options.wait) return;
		var from = {};
		var to = {};
		for (p in obj){
			from[p] = obj[p][0];
			to[p] = obj[p][1];
		}
		return this._start(from, to);
	},

	increase: function(){
		for (var p in this.now) this.setStyle(this.element, p, this.now[p]);
	}

});

//Transitions (c) 2003 Robert Penner (http://www.robertpenner.com/easing/), BSD License.

Fx.Transitions = {
	linear: function(t, b, c, d) { return c*t/d + b; },
	sineInOut: function(t, b, c, d) { return -c/2 * (Math.cos(Math.PI*t/d) - 1) + b; }
};

// start moo.fx.pack.js

//by Valerio Proietti (http://mad4milk.net). MIT-style license.
//moo.fx.pack.js - depends on prototype.js or prototype.lite.js + moo.fx.js
//version 2.0

Fx.Scroll = Class.create();
Fx.Scroll.prototype = Object.extend(new Fx.Base(), {

	initialize: function(el, options) {
		this.element = $(el);
		this.setOptions(options);
		this.element.style.overflow = 'hidden';
	},
	
	down: function(){
		return this.custom(this.element.scrollTop, this.element.scrollHeight-this.element.offsetHeight);
	},
	
	up: function(){
		return this.custom(this.element.scrollTop, 0);
	},

	increase: function(){
		this.element.scrollTop = this.now;
	}

});

//fx.Color, originally by Tom Jensen (http://neuemusic.com) MIT-style LICENSE.

Fx.Color = Class.create();
Fx.Color.prototype = Object.extend(new Fx.Base(), {
	
	initialize: function(el, property, options){
		this.element = $(el);
		this.setOptions(options);
		this.property = property.camelize();
		this.now = [];
	},

	custom: function(from, to){
		return this._start(from.hexToRgb(true), to.hexToRgb(true));
	},

	setNow: function(){
		[0,1,2].each(function(i){
			this.now[i] = Math.round(this.compute(this.from[i], this.to[i]));
		}.bind(this));
	},

	increase: function(){
		this.element.style[this.property] = "rgb("+this.now[0]+","+this.now[1]+","+this.now[2]+")";
	}

});

Object.extend(String.prototype, {

	rgbToHex: function(array){
		var rgb = this.match(new RegExp('([\\d]{1,3})', 'g'));
		if (rgb[3] == 0) return 'transparent';
		var hex = [];
		for (var i = 0; i < 3; i++){
			var bit = (rgb[i]-0).toString(16);
			hex.push(bit.length == 1 ? '0'+bit : bit);
		}
		var hexText = '#'+hex.join('');
		if (array) return hex;
		else return hexText;
	},

	hexToRgb: function(array){
		var hex = this.match(new RegExp('^[#]{0,1}([\\w]{1,2})([\\w]{1,2})([\\w]{1,2})$'));
		var rgb = [];
		for (var i = 1; i < hex.length; i++){
			if (hex[i].length == 1) hex[i] += hex[i];
			rgb.push(parseInt(hex[i], 16));
		}
		var rgbText = 'rgb('+rgb.join(',')+')';
		if (array) return rgb;
		else return rgbText;
	}	

});

// moo.fx.utils.js

//by Valerio Proietti (http://mad4milk.net). MIT-style license.
//moo.fx.utils.js - depends on prototype.js OR prototype.lite.js + moo.fx.js
//version 2.0

Fx.Height = Class.create();
Fx.Height.prototype = Object.extend(new Fx.Base(), {

	initialize: function(el, options){
		this.element = $(el);
		this.setOptions(options);
		this.element.style.overflow = 'hidden';
	},

	toggle: function(){
		if (this.element.offsetHeight > 0) return this.custom(this.element.offsetHeight, 0);
		else return this.custom(0, this.element.scrollHeight);
	},

	show: function(){
		return this.set(this.element.scrollHeight);
	},

	increase: function(){
		this.setStyle(this.element, 'height', this.now);
	}

});

Fx.Width = Class.create();
Fx.Width.prototype = Object.extend(new Fx.Base(), {

	initialize: function(el, options){
		this.element = $(el);
		this.setOptions(options);
		this.element.style.overflow = 'hidden';
		this.iniWidth = this.element.offsetWidth;
	},

	toggle: function(){
		if (this.element.offsetWidth > 0) return this.custom(this.element.offsetWidth, 0);
		else return this.custom(0, this.iniWidth);
	},

	show: function(){
		return this.set(this.iniWidth);
	},

	increase: function(){
		this.setStyle(this.element, 'width', this.now);
	}

});

Fx.Opacity = Class.create();
Fx.Opacity.prototype = Object.extend(new Fx.Base(), {

	initialize: function(el, options){
		this.element = $(el);
		this.setOptions(options);
		this.now = 1;
	},

	toggle: function(){
		if (this.now > 0) return this.custom(1, 0);
		else return this.custom(0, 1);
	},

	show: function(){
		return this.set(1);
	},
	
	increase: function(){
		this.setStyle(this.element, 'opacity', this.now);
	}

});

// accordion.js

//by Valerio Proietti (http://mad4milk.net). MIT-style license.
//accordion.js - depends on prototype.js or prototype.lite.js + moo.fx.js
//version 2.0

Fx.Accordion = Class.create();
Fx.Accordion.prototype = Object.extend(new Fx.Base(), {
	
	extendOptions: function(options){
		Object.extend(this.options, Object.extend({
			start: 'open-first',
			fixedHeight: false,
			fixedWidth: false,
			alwaysHide: false,
			wait: false,
			onActive: function(){},
			onBackground: function(){},
			height: true,
			opacity: true,
			width: false
		}, options || {}));
	},

	initialize: function(togglers, elements, options){
		this.now = {};
		this.elements = $A(elements);
		this.togglers = $A(togglers);
		this.setOptions(options);
		this.extendOptions(options);
		this.previousClick = 'nan';
		this.togglers.each(function(tog, i){
			if (tog.onclick) tog.prevClick = tog.onclick;
			else tog.prevClick = function(){};
			$(tog).onclick = function(){
				tog.prevClick();
				this.showThisHideOpen(i);
			}.bind(this);
		}.bind(this));
		this.h = {}; this.w = {}; this.o = {};
		this.elements.each(function(el, i){
			this.now[i+1] = {};
			el.style.height = '0';
			el.style.overflow = 'hidden';
		}.bind(this));
		switch(this.options.start){
			case 'first-open': this.elements[0].style.height = this.elements[0].scrollHeight+'px'; break;
			case 'open-first': this.showThisHideOpen(0); break;
		}
	},
	
	setNow: function(){
		for (var i in this.from){
			var iFrom = this.from[i];
			var iTo = this.to[i];
			var iNow = this.now[i] = {};
			for (var p in iFrom) iNow[p] = this.compute(iFrom[p], iTo[p]);
		}
	},

	custom: function(objObjs){
		if (this.timer && this.options.wait) return;
		var from = {};
		var to = {};
		for (var i in objObjs){
			var iProps = objObjs[i];
			var iFrom = from[i] = {};
			var iTo = to[i] = {};
			for (var prop in iProps){
				iFrom[prop] = iProps[prop][0];
				iTo[prop] = iProps[prop][1];
			}
		}
		return this._start(from, to);
	},

	hideThis: function(i){
		if (this.options.height) this.h = {'height': [this.elements[i].offsetHeight, 0]};
		if (this.options.width) this.w = {'width': [this.elements[i].offsetWidth, 0]};
		if (this.options.opacity) this.o = {'opacity': [this.now[i+1]['opacity'] || 1, 0]};
	},

	showThis: function(i){
		if (this.options.height) this.h = {'height': [this.elements[i].offsetHeight, this.options.fixedHeight || this.elements[i].scrollHeight]};
		if (this.options.width) this.w = {'width': [this.elements[i].offsetWidth, this.options.fixedWidth || this.elements[i].scrollWidth]};
		if (this.options.opacity) this.o = {'opacity': [this.now[i+1]['opacity'] || 0, 1]};
	},

	showThisHideOpen: function(iToShow){
		if (iToShow != this.previousClick || this.options.alwaysHide){
			this.previousClick = iToShow;
			var objObjs = {};
			var err = false;
			var madeInactive = false;
			this.elements.each(function(el, i){
				this.now[i] = this.now[i] || {};
				if (i != iToShow){
					this.hideThis(i);
				} else if (this.options.alwaysHide){
					if (el.offsetHeight == el.scrollHeight){
						this.hideThis(i);
						madeInactive = true;
					} else if (el.offsetHeight == 0){
						this.showThis(i);
					} else {
						err = true;
					}
				} else if (this.options.wait && this.timer){
					this.previousClick = 'nan';
					err = true;
				} else {
					this.showThis(i);
				}
				objObjs[i+1] = Object.extend(this.h, Object.extend(this.o, this.w));
			}.bind(this));
			if (err) return;
			if (!madeInactive) this.options.onActive.call(this, this.togglers[iToShow], iToShow);
			this.togglers.each(function(tog, i){
				if (i != iToShow || madeInactive) this.options.onBackground.call(this, tog, i);
			}.bind(this));
			return this.custom(objObjs);
		}
	},
	
	increase: function(){
		for (var i in this.now){
			var iNow = this.now[i];
			for (var p in iNow) this.setStyle(this.elements[parseInt(i)-1], p, iNow[p]);
		}
	}

});

// moo.fx.transitions.js

//moo.fx.transitions.js - depends on prototype.js or prototype.lite.js + moo.fx.js
//Author: Robert Penner, <http://www.robertpenner.com/easing/>, modified to be used with mootools.
//License: Easing Equations v1.5, (c) 2003 Robert Penner, all rights reserved. Open Source BSD License.

Fx.Transitions = {

	linear: function(t, b, c, d){
		return c*t/d + b;
	},

	quadIn: function(t, b, c, d){
		return c*(t/=d)*t + b;
	},

	quadOut: function(t, b, c, d){
		return -c *(t/=d)*(t-2) + b;
	},

	quadInOut: function(t, b, c, d){
		if ((t/=d/2) < 1) return c/2*t*t + b;
		return -c/2 * ((--t)*(t-2) - 1) + b;
	},

	cubicIn: function(t, b, c, d){
		return c*(t/=d)*t*t + b;
	},

	cubicOut: function(t, b, c, d){
		return c*((t=t/d-1)*t*t + 1) + b;
	},

	cubicInOut: function(t, b, c, d){
		if ((t/=d/2) < 1) return c/2*t*t*t + b;
		return c/2*((t-=2)*t*t + 2) + b;
	},

	quartIn: function(t, b, c, d){
		return c*(t/=d)*t*t*t + b;
	},

	quartOut: function(t, b, c, d){
		return -c * ((t=t/d-1)*t*t*t - 1) + b;
	},

	quartInOut: function(t, b, c, d){
		if ((t/=d/2) < 1) return c/2*t*t*t*t + b;
		return -c/2 * ((t-=2)*t*t*t - 2) + b;
	},

	quintIn: function(t, b, c, d){
		return c*(t/=d)*t*t*t*t + b;
	},

	quintOut: function(t, b, c, d){
		return c*((t=t/d-1)*t*t*t*t + 1) + b;
	},

	quintInOut: function(t, b, c, d){
		if ((t/=d/2) < 1) return c/2*t*t*t*t*t + b;
		return c/2*((t-=2)*t*t*t*t + 2) + b;
	},

	sineIn: function(t, b, c, d){
		return -c * Math.cos(t/d * (Math.PI/2)) + c + b;
	},

	sineOut: function(t, b, c, d){
		return c * Math.sin(t/d * (Math.PI/2)) + b;
	},

	sineInOut: function(t, b, c, d){
		return -c/2 * (Math.cos(Math.PI*t/d) - 1) + b;
	},

	expoIn: function(t, b, c, d){
		return (t==0) ? b : c * Math.pow(2, 10 * (t/d - 1)) + b;
	},

	expoOut: function(t, b, c, d){
		return (t==d) ? b+c : c * (-Math.pow(2, -10 * t/d) + 1) + b;
	},

	expoInOut: function(t, b, c, d){
		if (t==0) return b;
		if (t==d) return b+c;
		if ((t/=d/2) < 1) return c/2 * Math.pow(2, 10 * (t - 1)) + b;
		return c/2 * (-Math.pow(2, -10 * --t) + 2) + b;
	},

	circIn: function(t, b, c, d){
		return -c * (Math.sqrt(1 - (t/=d)*t) - 1) + b;
	},

	circOut: function(t, b, c, d){
		return c * Math.sqrt(1 - (t=t/d-1)*t) + b;
	},

	circInOut: function(t, b, c, d){
		if ((t/=d/2) < 1) return -c/2 * (Math.sqrt(1 - t*t) - 1) + b;
		return c/2 * (Math.sqrt(1 - (t-=2)*t) + 1) + b;
	},

	elasticIn: function(t, b, c, d, a, p){
		if (t==0) return b; if ((t/=d)==1) return b+c; if (!p) p=d*.3; if (!a) a = 1;
		if (a < Math.abs(c)){ a=c; var s=p/4; }
		else var s = p/(2*Math.PI) * Math.asin(c/a);
		return -(a*Math.pow(2,10*(t-=1)) * Math.sin( (t*d-s)*(2*Math.PI)/p )) + b;
	},

	elasticOut: function(t, b, c, d, a, p){
		if (t==0) return b; if ((t/=d)==1) return b+c; if (!p) p=d*.3; if (!a) a = 1;
		if (a < Math.abs(c)){ a=c; var s=p/4; }
		else var s = p/(2*Math.PI) * Math.asin(c/a);
		return a*Math.pow(2,-10*t) * Math.sin( (t*d-s)*(2*Math.PI)/p ) + c + b;
	},

	elasticInOut: function(t, b, c, d, a, p){
		if (t==0) return b; if ((t/=d/2)==2) return b+c; if (!p) p=d*(.3*1.5); if (!a) a = 1;
		if (a < Math.abs(c)){ a=c; var s=p/4; }
		else var s = p/(2*Math.PI) * Math.asin(c/a);
		if (t < 1) return -.5*(a*Math.pow(2,10*(t-=1)) * Math.sin( (t*d-s)*(2*Math.PI)/p )) + b;
		return a*Math.pow(2,-10*(t-=1)) * Math.sin( (t*d-s)*(2*Math.PI)/p )*.5 + c + b;
	},

	backIn: function(t, b, c, d, s){
		if (!s) s = 1.70158;
		return c*(t/=d)*t*((s+1)*t - s) + b;
	},

	backOut: function(t, b, c, d, s){
		if (!s) s = 1.70158;
		return c*((t=t/d-1)*t*((s+1)*t + s) + 1) + b;
	},

	backInOut: function(t, b, c, d, s){
		if (!s) s = 1.70158;
		if ((t/=d/2) < 1) return c/2*(t*t*(((s*=(1.525))+1)*t - s)) + b;
		return c/2*((t-=2)*t*(((s*=(1.525))+1)*t + s) + 2) + b;
	},

	bounceIn: function(t, b, c, d){
		return c - Fx.Transitions.bounceOut (d-t, 0, c, d) + b;
	},

	bounceOut: function(t, b, c, d){
		if ((t/=d) < (1/2.75)){
			return c*(7.5625*t*t) + b;
		} else if (t < (2/2.75)){
			return c*(7.5625*(t-=(1.5/2.75))*t + .75) + b;
		} else if (t < (2.5/2.75)){
			return c*(7.5625*(t-=(2.25/2.75))*t + .9375) + b;
		} else {
			return c*(7.5625*(t-=(2.625/2.75))*t + .984375) + b;
		}
	},

	bounceInOut: function(t, b, c, d){
		if (t < d/2) return Fx.Transitions.bounceIn(t*2, 0, c, d) * .5 + b;
		return Fx.Transitions.bounceOut(t*2-d, 0, c, d) * .5 + c*.5 + b;
	}

};